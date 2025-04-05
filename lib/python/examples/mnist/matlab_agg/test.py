import logging
import warnings
import numpy as np
import pandas as pd
import os
import random # Added for random settings generation
from datetime import datetime
from flame.mode.message import MessageType
from flame.config import Config
from flame.dataset import Dataset
from flame.mode.horizontal.top_aggregator import TopAggregator
# Attempt to import PySpark and ALS
# try:
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
PYSPARK_AVAILABLE = True
# except ImportError as e:
#     PYSPARK_AVAILABLE = False
#     SparkSession = None # Define dummy classes/vars if import fails
#     ALS = None
#     StructType = None
#     StructField = None
#     IntegerType = None
#     FloatType = None
#     logging.warning(f"PySpark import failed: {e}. Matrix completion features will be disabled.")

os.environ['JAVA_OPTS'] = '-Djava.security.manager=allow'

'''
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
export PATH=$JAVA_HOME/bin:$PATH
'''

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

E_RMS = "e_rms"
PRINT_TIME = "time"
# --- Constants for operational phases ---
INITIAL_RANDOM_ROUNDS = 18 # Number of rounds for initial random exploration
RETRAIN_INTERVAL = 5 # Retrain model every N rounds after initial phase

class PrinterAggregator(TopAggregator):
    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.model = None # Will hold the MatrixCompletionModel instance
        self.dataset = None # Not used in this implementation logic

        # --- Default Printer Settings (from config) ---
        # Use getattr() for safer access to hyperparameters on the config object
        self.default_feedrate = float(getattr(self.config.hyperparameters, "feedrate", 100.0))
        self.default_acc = float(getattr(self.config.hyperparameters, "acceleration", 5000.0))

        # --- Metrics and Settings Storage ---
        self.trainer_metrics = {} # Stores received metrics {end: {E_RMS: val, PRINT_TIME: val, 'history': []}}
        # Stores settings RECOMMENDED for the NEXT round (used in optimization phase)
        self.trainer_settings = {} # {end: {"feedrate": val, "acceleration": val}}
        # Stores settings SENT in the CURRENT round (needed for random phase data mapping)
        self.sent_settings_this_round = {} # {end: {"feedrate": val, "acceleration": val}}

        self.device = None # Not used in this logic
        self.test_loader = None # Not used in this logic

        # --- Printer History Tracking ---
        self.printer_history = {} # {end: [ { E_RMS: ..., PRINT_TIME: ..., etc }, ... ] }

        # --- Matrix Completion Attributes ---
        self.spark = None
        self.als_model = None # ALS model configuration
        self.als_trained_model = None # Trained ALS model instance
        self.train_data = None # Spark DataFrame for training data (PrinterId, ConditionId, Loss)
        self.printer_id_map = {} # Maps endpoint ID (str) to internal PrinterId (int)
        self.next_printer_id = 0
        self.condition_settings = [] # List mapping ConditionId (int index) to [feed, acc] (list)
        self.condition_id_map = {} # Maps (feed, acc) tuple to ConditionId (int)
        self.new_data_points = [] # Stores [(PrinterId, ConditionId, Loss)] tuples for next training batch
        self.current_end = None # Tracks the endpoint being processed in evaluate()
        self._rounds = int(getattr(self.config.hyperparameters, "rounds", 30)) # Total rounds for the process

        # --- Configurable Grid Parameters ---
        # Read from config using getattr() or use defaults
        self.feedrate_min = float(getattr(self.config.hyperparameters, "feedrate_min", 50.0))
        self.feedrate_max = float(getattr(self.config.hyperparameters, "feedrate_max", 150.0))
        self.feedrate_increment = float(getattr(self.config.hyperparameters, "feedrate_increment", 25.0))
        self.acceleration_min = float(getattr(self.config.hyperparameters, "acceleration_min", 4000.0))
        self.acceleration_max = float(getattr(self.config.hyperparameters, "acceleration_max", 7000.0))
        self.acceleration_increment = float(getattr(self.config.hyperparameters, "acceleration_increment", 500.0))
        self.col_size = 0 # Number of conditions, calculated in initialize

        # --- Column Names for Spark DataFrame ---
        self.COL_PRINTER = "PrinterId"
        self.COL_CONDITION = "ConditionId"
        self.COL_LOSS = "Loss"
        self.schema = None # Defined in initialize if PySpark is available

        # --- State Flag ---
        self.initial_training_done = False # Tracks if the first training cycle is complete

    def initialize(self):
        """Initialize role, Spark session, condition grid, and ALS model components."""
        # --- PySpark Availability Check ---
        # if not PYSPARK_AVAILABLE:
        #     logger.error("PySpark not found. Cannot initialize matrix completion model.")
        #     # Define a dummy model that always returns defaults
        #     self.model = lambda err, time: (self.default_feedrate, self.default_acc)
        #     # Define dummy _find_closest_condition to avoid errors if called
        #     self._find_closest_condition = lambda f, a: 0
        #     return # Stop initialization here

        try:
            logger.info("Initializing Spark session and ALS model components")

            # --- Initialize Spark Session ---
            # Use getattr for spark memory config, defaulting to "4g" as in reference
            spark_mem = getattr(self.config.hyperparameters, "spark_memory", "4g")
            self.spark = SparkSession.builder.appName("ALS 3D Printer Optimization") \
                .config("spark.memory", spark_mem) \
                .config("spark.sql.analyzer.failAmbiguousSelfJoin", "false") \
                .getOrCreate()
            logger.info(f"Spark session initialized. Spark version: {self.spark.version}. Configured spark.memory: {spark_mem}")

            # --- Define Schema for Training Data ---
            self.schema = StructType([
                StructField(self.COL_PRINTER, IntegerType(), False),
                StructField(self.COL_CONDITION, IntegerType(), False),
                StructField(self.COL_LOSS, FloatType(), False)
            ])

            # --- Initialize Empty Training Data ---
            # Replaces loading from 'large_J_mat.csv'
            self.train_data = self.spark.createDataFrame([], self.schema)
            logger.info("Initialized with an empty training dataset (Spark DataFrame).")

            # --- Create Condition Settings Grid (Items) ---
            # Validate increments to avoid infinite loops or zero division
            if self.feedrate_increment <= 0 or self.acceleration_increment <= 0:
                 raise ValueError("Feedrate and Acceleration increments must be positive.")
            if self.feedrate_min > self.feedrate_max or self.acceleration_min > self.acceleration_max:
                 raise ValueError("Min values cannot be greater than Max values for feedrate/acceleration.")

            # Calculate counts ensuring at least one value if min==max
            feedrate_count = int(np.floor((self.feedrate_max - self.feedrate_min) / self.feedrate_increment)) + 1 if self.feedrate_max >= self.feedrate_min else 0
            acceleration_count = int(np.floor((self.acceleration_max - self.acceleration_min) / self.acceleration_increment)) + 1 if self.acceleration_max >= self.acceleration_min else 0
            self.col_size = feedrate_count * acceleration_count # Total number of conditions (items)

            if self.col_size == 0:
                 raise ValueError("Grid parameters result in zero conditions. Check min/max/increment values.")

            # Generate the grid values
            feedrate_values = np.linspace(self.feedrate_min, self.feedrate_max, feedrate_count) if feedrate_count > 0 else []
            acceleration_values = np.linspace(self.acceleration_min, self.acceleration_max, acceleration_count) if acceleration_count > 0 else []

            condition_id = 0
            self.condition_settings = [] # Ensure it's empty before populating
            self.condition_id_map = {}
            # Iterate column-major (feedrate changes faster within each acceleration level)
            for acc in acceleration_values:
                for feed in feedrate_values:
                    # Ensure values are standard Python floats
                    feed_f = float(feed)
                    acc_f = float(acc)
                    self.condition_settings.append([feed_f, acc_f])
                    self.condition_id_map[(feed_f, acc_f)] = condition_id
                    condition_id += 1
            logger.info(f"Generated {len(self.condition_settings)} unique conditions (col_size = {self.col_size}) from grid parameters.")
            # --- End Condition Settings Grid ---

            # --- Configure ALS Model ---
            # Read hyperparameters from config using getattr() or use defaults
            als_rank = int(getattr(self.config.hyperparameters, "als_rank", 3)) # Defaulting rank to 3 now
            als_maxIter = int(getattr(self.config.hyperparameters, "als_maxIter", 15))
            als_regParam = float(getattr(self.config.hyperparameters, "als_regParam", 0.1)) # Defaulting regParam to 0.1 now

            self.als_model = ALS(
                rank=als_rank,
                maxIter=als_maxIter,
                regParam=als_regParam,
                implicitPrefs=False, # We have explicit 'loss' ratings
                coldStartStrategy='drop', # Drop users/items not in training data during prediction
                nonnegative=True, # Loss should be non-negative
                seed=42,
                userCol=self.COL_PRINTER,
                itemCol=self.COL_CONDITION,
                ratingCol=self.COL_LOSS
            )
            logger.info(f"ALS model configured: rank={als_rank}, maxIter={als_maxIter}, regParam={als_regParam}")

            # --- Define Model Wrapper Class (as in reference) ---
            class MatrixCompletionModel:
                def __init__(self, aggregator):
                    self.aggregator = aggregator

                def __call__(self, err, print_time):
                    """
                    Return optimal feedrate/acceleration based on error, print time, and trained model.
                    Also adds the new data point for the *evaluated* setting during optimization phase.
                    """
                    current_end = self.aggregator.current_end # Get current printer being evaluated

                    # --- Pre-check: Model Readiness ---
                    # This should only be called during the optimization phase AFTER initial training
                    if not self.aggregator.initial_training_done or not self.aggregator.als_trained_model:
                        logger.warning(f"ALS model not ready for prediction for {current_end}. Returning defaults.")
                        return self.aggregator.default_feedrate, self.aggregator.default_acc

                    # --- Get Printer ID (User) ---
                    printer_idx = self.aggregator._get_or_assign_printer_id(current_end)

                    # --- Calculate Loss & Add Data Point for *Current* Settings ---
                    # This step happens based on the settings the printer *just ran*
                    # We need the settings that were *sent* in the previous round (which led to current err, print_time)
                    # These are stored in self.aggregator.trainer_settings[current_end] if available
                    prev_settings = self.aggregator.trainer_settings.get(current_end)
                    if prev_settings:
                        prev_feedrate = prev_settings.get("feedrate", self.aggregator.default_feedrate)
                        prev_acceleration = prev_settings.get("acceleration", self.aggregator.default_acc)
                    else:
                        # If no previous settings recorded (e.g., first optimization round), use defaults
                        prev_feedrate = self.aggregator.default_feedrate
                        prev_acceleration = self.aggregator.default_acc
                        logger.warning(f"No previous settings found for {current_end} in optimization phase. Using defaults for data point.")

                    # Find condition ID for the settings the printer *ran*
                    ran_condition_id = self.aggregator._find_closest_condition(prev_feedrate, prev_acceleration)

                    # Calculate combined loss for the settings the printer *ran*
                    # Normalize based on expected ranges (tune these if needed)
                    normalized_err = err / 0.1 if err is not None and err > 1e-6 else (10.0 if err is not None else 20.0) # High penalty for low/None error
                    normalized_time = (print_time / 60.0) if print_time is not None and print_time > 0 else 10.0 # Normalize to minutes, penalty if None/0

                    # Example weighting: Error is twice as important as time
                    combined_loss = 2.0 * normalized_err + 1.0 * normalized_time
                    combined_loss = max(0.0, combined_loss) # Ensure non-negative

                    # Add this observation (printer_idx, ran_condition_id, combined_loss) to list for next training
                    self.aggregator.new_data_points.append([printer_idx, ran_condition_id, float(combined_loss)])
                    logger.debug(f"Optimization Phase: Added data point for {current_end} (Printer ID {printer_idx}): "
                                 f"Ran Condition={ran_condition_id} ({prev_feedrate:.1f}f/{prev_acceleration:.0f}a), "
                                 f"RMS={err:.4f}, Time={print_time:.2f}s -> Loss={combined_loss:.4f}")
                    # --- End Data Point Addition ---


                    # --- Generate Predictions for Recommendation ---
                    # Create a DataFrame with all conditions for the current printer
                    conditions_to_predict = [(printer_idx, i) for i in range(self.aggregator.col_size)]
                    predict_df = pd.DataFrame(conditions_to_predict, columns=[self.aggregator.COL_PRINTER, self.aggregator.COL_CONDITION])

                    try:
                        predict_spark_df = self.aggregator.spark.createDataFrame(predict_df)

                        # Make predictions using the trained ALS model
                        predictions = self.aggregator.als_trained_model.transform(predict_spark_df)

                        # Find the condition with the minimum predicted loss (lower is better)
                        valid_predictions = predictions.filter(predictions["prediction"].isNotNull())
                        if valid_predictions.count() == 0:
                             logger.warning(f"No valid predictions generated for printer {printer_idx} ({current_end}). Returning defaults.")
                             return self.aggregator.default_feedrate, self.aggregator.default_acc

                        min_row = valid_predictions.orderBy("prediction").first()

                        if min_row:
                            best_condition_id = min_row[self.aggregator.COL_CONDITION]
                            predicted_loss = min_row["prediction"]

                            # Get the settings for this best predicted condition
                            rec_feedrate, rec_acceleration = self.aggregator.condition_settings[best_condition_id]
                            logger.info(f"Recommending settings for {current_end} (Printer ID {printer_idx}): "
                                        f"Feedrate={rec_feedrate:.2f}, Acceleration={rec_acceleration:.2f} "
                                        f"(Condition ID {best_condition_id}, Predicted Loss: {predicted_loss:.4f})")
                            return float(rec_feedrate), float(rec_acceleration)
                        else:
                            logger.warning(f"Could not find minimum prediction for {current_end}. Returning defaults.")
                            return self.aggregator.default_feedrate, self.aggregator.default_acc

                    except Exception as e:
                        # Catch potential Spark errors during prediction
                        logger.error(f"Error during prediction transform/orderBy for {current_end}: {e}", exc_info=True)
                        return self.aggregator.default_feedrate, self.aggregator.default_acc
            # --- End Model Wrapper Class ---

            # --- Create Model Instance ---
            self.model = MatrixCompletionModel(self)
            logger.info("Matrix completion model wrapper initialized successfully.")

        except ImportError:
             # This case is handled by the PYSPARK_AVAILABLE check at the start
             pass # Already logged warning
        except Exception as e:
            logger.error(f"Critical error initializing Spark or ALS components: {e}", exc_info=True)
            logger.warning("Falling back to default settings behavior.")
            # PYSPARK_AVAILABLE = False # Disable pyspark features
            self.model = lambda err, time: (self.default_feedrate, self.default_acc)
            self._find_closest_condition = lambda f, a: 0


    def get(self, tag: str) -> None:
        """Get data (metrics) from remote printer nodes (trainers)."""
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"Channel not found for tag {tag} in get")
            return

        # Receive local metrics from printer nodes
        for msg, metadata in channel.recv_fifo(channel.ends()):
            end, timestamp = metadata
            if not msg:
                logger.debug(f"No data received from {end} in round {self._round}; skipping.")
                continue

            logger.debug(f"Received data from {end} in round {self._round}: {msg}")
            channel.set_end_property(end, "round_end_time", (self._round, timestamp))

            # Extract metrics, handle missing keys gracefully
            e_rms = msg.get(E_RMS) # Use .get for safety
            print_time = msg.get(PRINT_TIME)

            # Initialize metrics storage for the end if first time
            if end not in self.trainer_metrics:
                self.trainer_metrics[end] = {} # Store metrics directly
            if end not in self.printer_history: # Use separate history dict
                 self.printer_history[end] = []

            # Store current metrics for evaluation in this round
            self.trainer_metrics[end][E_RMS] = e_rms
            self.trainer_metrics[end][PRINT_TIME] = print_time

            # Add current metrics and *sent* settings to history (if valid metrics)
            if e_rms is not None and print_time is not None:
                 # Retrieve settings that were *sent* in this round for history logging
                sent_settings = self.sent_settings_this_round.get(end, {}) # Get from dict used in put()
                sent_feedrate = sent_settings.get("feedrate", None)
                sent_acceleration = sent_settings.get("acceleration", None)

                history_entry = {
                    E_RMS: e_rms,
                    PRINT_TIME: print_time,
                    "feedrate_sent": sent_feedrate,
                    "acceleration_sent": sent_acceleration,
                    "round": self._round,
                    "timestamp": timestamp
                }
                self.printer_history[end].append(history_entry)
            else:
                logger.warning(f"Incomplete metrics received from {end} in round {self._round}. RMS: {e_rms}, Time: {print_time}")


    def put(self, tag: str) -> None:
        """Set data (settings) to remote printer nodes (trainers)."""
        self.dist_tag = tag # Not used in reference, maybe specific to FLAME setup
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"Channel not found for tag {tag} in put")
            return

        # Wait for at least one peer to join this channel (as in reference)
        channel.await_join()

        # Alternative: Check if ends exist and proceed if they do
        if not channel.ends():
             logger.warning(f"No ends connected on channel {tag} in round {self._round}. Cannot send settings.")
             return

        selected_ends = channel.ends()
        self.sent_settings_this_round.clear() # Clear settings sent in the previous round

        # Send out settings to printer nodes
        for end in selected_ends:
            feedrate_to_send = self.default_feedrate
            acceleration_to_send = self.default_acc

            # --- Logic for determining settings based on round ---
            if self._round < INITIAL_RANDOM_ROUNDS:
                # Phase 1: Send random settings for exploration
                feedrate_to_send = random.uniform(self.feedrate_min, self.feedrate_max)
                acceleration_to_send = random.uniform(self.acceleration_min, self.acceleration_max)
                logger.info(f"[Round {self._round}/{self._rounds} - Random Phase] Sending random settings to {end}: "
                            f"Feedrate={feedrate_to_send:.2f}, Acceleration={acceleration_to_send:.2f}")
            else: # Optimization Phase (or fallback if model not ready)
                # Retrieve recommended settings from self.trainer_settings (populated by evaluate)
                if end in self.trainer_settings:
                    recommended = self.trainer_settings[end]
                    feedrate_to_send = recommended.get("feedrate", self.default_feedrate)
                    acceleration_to_send = recommended.get("acceleration", self.default_acc)
                    logger.info(f"[Round {self._round}/{self._rounds} - Optimization Phase] Sending recommended settings to {end}: "
                                f"Feedrate={feedrate_to_send:.2f}, Acceleration={acceleration_to_send:.2f}")
                else:
                    # If no recommendation exists (e.g., first optimization round, or error), send defaults
                    logger.warning(f"No recommended settings found for {end} in round {self._round}. Sending defaults.")
                    feedrate_to_send = self.default_feedrate
                    acceleration_to_send = self.default_acc
            # --- End settings determination logic ---

            # Store the settings being sent *in this round* for later mapping in evaluate/get
            self.sent_settings_this_round[end] = {
                "feedrate": feedrate_to_send,
                "acceleration": acceleration_to_send
            }

            # Send the message
            message_payload = {
                MessageType.ROUND: self._round,
                "feedrate": feedrate_to_send,
                "acceleration": acceleration_to_send,
            }
            channel.send(end, message_payload)

            # Register round start time (as in reference)
            channel.set_end_property(end, "round_start_time", (self._round, datetime.now()))

            # Log sent settings (moved from reference loop body to here)
            # logger.info(f"Sent settings to {end}: feedrate={feedrate_to_send}, acceleration={acceleration_to_send}")


    def load_data(self) -> None:
        """Load data (not used in this version as data is generated/collected)."""
        logger.debug("load_data() called, but data is generated dynamically.")
        pass

    def train(self) -> None:
        """Train or retrain the ALS model with accumulated data points."""
        # --- Pre-checks ---
        # if not PYSPARK_AVAILABLE:
        #     logger.warning("Cannot train model: PySpark is not available.")
        #     return
        if not self.spark:
             logger.error("Spark session not initialized. Cannot train model.")
             return
        if not self.new_data_points:
            logger.info(f"No new data points collected since last training in round {self._round}. Skipping training.")
            return

        logger.info(f"Starting model training with {len(self.new_data_points)} new data points...")

        try:
            # --- Prepare New Data ---
            # Convert new data points (list of lists) to a Pandas DataFrame
            new_data_pd = pd.DataFrame(self.new_data_points,
                                       columns=[self.COL_PRINTER, self.COL_CONDITION, self.COL_LOSS])

            # Convert Pandas DataFrame to Spark DataFrame using the defined schema
            new_spark_df = self.spark.createDataFrame(new_data_pd, schema=self.schema)

            # --- Update Training Data ---
            # Add new data to the existing training data Spark DataFrame
            # Use distinct() to avoid duplicates if the same printer/condition pair is added again
            # This is crucial if multiple evaluations happen before training
            self.train_data = self.train_data.union(new_spark_df).distinct()

            # Cache the training data for potentially faster ALS iterations
            self.train_data.cache()
            current_data_size = self.train_data.count()
            logger.info(f"Total training data size after adding new points: {current_data_size} distinct points.")

            # --- Train ALS Model ---
            # Check if there's enough data (ALS requires non-empty data)
            if current_data_size == 0:
                 logger.warning("Training data is empty. Skipping ALS fitting.")
                 self.train_data.unpersist() # Unpersist if we skip fitting
                 return

            # Re-use the configured self.als_model instance
            self.als_trained_model = self.als_model.fit(self.train_data)

            # Unpersist the training data after training to free up memory
            self.train_data.unpersist()

            logger.info(f"ALS model training completed successfully for round {self._round}.")

            # --- Post-Training Actions ---
            # Store training history (optional, as in reference)
            if not hasattr(self, 'training_history'):
                self.training_history = []
            self.training_history.append({
                'round': self._round,
                'timestamp': datetime.now(),
                'new_points_added': len(self.new_data_points), # Points attempted to add
                'total_points_trained': current_data_size # Actual distinct points used
            })

            # Clear the list of new data points as they are now incorporated
            self.new_data_points = []
            self.initial_training_done = True # Mark that at least one training cycle is complete

        except Exception as e:
            logger.error(f"Error during model training in round {self._round}: {e}", exc_info=True)
            # Optionally unpersist data even if training failed
            if self.train_data and self.train_data.is_cached:
                 self.train_data.unpersist()


    def evaluate(self) -> None:
        """
        Evaluate metrics: Collect data points (Phase 1) or generate recommendations (Phase 2).
        Triggers model training at appropriate rounds.
        """
        logger.info(f"Entering evaluate phase for round {self._round}...")
        # Note: Recommendations for the *next* round are stored in self.trainer_settings

        # Process metrics received from each printer that sent data
        printers_with_metrics = list(self.trainer_metrics.keys())
        logger.debug(f"Evaluating metrics for printers: {printers_with_metrics}")

        for end in printers_with_metrics:
            # --- Check for complete metrics ---
            metrics = self.trainer_metrics.get(end, {})
            err = metrics.get(E_RMS)
            print_time = metrics.get(PRINT_TIME)

            if err is None or print_time is None:
                logger.warning(f"Incomplete or missing metrics for {end} in round {self._round} (RMS: {err}, Time: {print_time}). Skipping evaluation.")
                continue

            # --- Set current printer context ---
            self.current_end = end

            # --- Phase-Dependent Logic ---
            if self._round < INITIAL_RANDOM_ROUNDS:
                # --- Phase 1: Random Exploration Data Collection ---
                # Retrieve the random settings that were *sent* in this round
                sent_settings = self.sent_settings_this_round.get(end)
                if not sent_settings:
                     logger.warning(f"No sent settings found for {end} in random phase round {self._round}. Cannot add data point.")
                     continue

                sent_feedrate = sent_settings["feedrate"]
                sent_acceleration = sent_settings["acceleration"]

                # Find the condition ID corresponding to the *sent* random settings
                condition_id = self._find_closest_condition(sent_feedrate, sent_acceleration)

                # Calculate combined loss based on received metrics
                normalized_err = err / 0.1 if err is not None and err > 1e-6 else (10.0 if err is not None else 20.0)
                normalized_time = (print_time / 60.0) if print_time is not None and print_time > 0 else 10.0
                combined_loss = 2.0 * normalized_err + 1.0 * normalized_time
                combined_loss = max(0.0, combined_loss) # Ensure non-negative

                # Get or assign printer ID
                printer_idx = self._get_or_assign_printer_id(end)

                # Add this observation to our list for the *initial* training batch
                self.new_data_points.append([printer_idx, condition_id, float(combined_loss)])
                logger.debug(f"Random Phase: Added data point for {end} (Printer ID {printer_idx}): "
                             f"Sent Condition={condition_id} ({sent_feedrate:.1f}f/{sent_acceleration:.0f}a), "
                             f"RMS={err:.4f}, Time={print_time:.2f}s -> Loss={combined_loss:.4f}")
                # Do NOT generate recommendations or update self.trainer_settings in this phase

            else:
                # --- Phase 2: Optimization Recommendation Generation ---
                if not self.initial_training_done or not self.model:
                     logger.warning(f"Model not ready for optimization phase for {end} round {self._round}. Using defaults.")
                     rec_feedrate = self.default_feedrate
                     rec_acceleration = self.default_acc
                else:
                     try:
                         # Use the model wrapper to get recommended settings for the *next* round
                         # The wrapper's __call__ method ALSO adds the data point for the *current* metrics/settings
                         rec_feedrate, rec_acceleration = self.model(err, print_time)
                     except Exception as e:
                         logger.error(f"Error calling model for recommendation for {end}: {e}", exc_info=True)
                         rec_feedrate = self.default_feedrate
                         rec_acceleration = self.default_acc

                # Update self.trainer_settings with the recommendation for the *next* round's put()
                if end not in self.trainer_settings:
                    self.trainer_settings[end] = {}
                self.trainer_settings[end]["feedrate"] = rec_feedrate
                self.trainer_settings[end]["acceleration"] = rec_acceleration

                # Update history with recommended settings (optional, similar to reference)
                if end in self.printer_history and self.printer_history[end]:
                    self.printer_history[end][-1]["feedrate_recommended"] = rec_feedrate
                    self.printer_history[end][-1]["acceleration_recommended"] = rec_acceleration

                # logger.info(f"Updated recommended settings for {end} for next round: feedrate={rec_feedrate}, acceleration={rec_acceleration}")

        # --- Clear metrics for the round after processing all ends ---
        self.trainer_metrics.clear()

        # --- Trigger Training Logic ---
        # Trigger initial training at the end of the random phase
        if self._round == INITIAL_RANDOM_ROUNDS - 1:
            logger.info(f"End of random exploration phase (Round {self._round}). Triggering initial model training.")
            self.train()
        # Trigger periodic retraining after the initial phase, only if new data exists
        elif self._round >= INITIAL_RANDOM_ROUNDS and \
             (self._round - (INITIAL_RANDOM_ROUNDS - 1)) % RETRAIN_INTERVAL == 0:
             if self.new_data_points: # Only train if there's new data
                 logger.info(f"Reached retraining interval (Round {self._round}) with new data. Triggering model retraining.")
                 self.train()
             else:
                 logger.info(f"Reached retraining interval (Round {self._round}) but no new data points. Skipping retraining.")
        # --- End Trigger Training ---

        logger.info(f"Finished evaluate phase for round {self._round}.")


    def _find_closest_condition(self, feedrate, acceleration):
        """Find the condition ID in the predefined grid closest to the given settings."""
        if not self.condition_settings: # Should not happen if initialize worked
             logger.error("Condition settings grid is empty. Cannot find closest condition.")
             return 0 # Return a default index

        min_distance_sq = float('inf')
        closest_id = 0

        # Normalize the ranges for distance calculation only if range > 0
        feedrate_range = (self.feedrate_max - self.feedrate_min) if self.feedrate_max > self.feedrate_min else 1.0
        acc_range = (self.acceleration_max - self.acceleration_min) if self.acceleration_max > self.acceleration_min else 1.0
        # Avoid division by zero if range is zero (min == max)
        feedrate_range = max(feedrate_range, 1e-6)
        acc_range = max(acc_range, 1e-6)


        for idx, (cond_feedrate, cond_acc) in enumerate(self.condition_settings):
            # Calculate squared Euclidean distance with normalization
            normalized_feedrate_diff = (feedrate - cond_feedrate) / feedrate_range
            normalized_acc_diff = (acceleration - cond_acc) / acc_range
            distance_sq = normalized_feedrate_diff**2 + normalized_acc_diff**2

            if distance_sq < min_distance_sq:
                min_distance_sq = distance_sq
                closest_id = idx

        return closest_id

    def _get_or_assign_printer_id(self, end_id):
        """Gets the internal integer ID for a printer endpoint, assigning a new one if necessary."""
        if end_id not in self.printer_id_map:
            new_id = self.next_printer_id
            self.printer_id_map[end_id] = new_id
            self.next_printer_id += 1
            logger.info(f"Assigned new printer ID {new_id} to endpoint {end_id}")
            # Note: If a new printer joins mid-optimization, ALS coldStartStrategy ('drop') will handle it.
            # The model won't predict for it until it appears in retrained data.
        return self.printer_id_map[end_id]


# Main execution block (Standard FLAME execution)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Printer Aggregator for federated learning')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    a = PrinterAggregator(config)
    a.compose()
    a.run()