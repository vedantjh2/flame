import logging
import warnings
import numpy as np
import pandas as pd
import os
from datetime import datetime
from flame.mode.message import MessageType
from flame.config import Config
from flame.dataset import Dataset
from flame.mode.horizontal.top_aggregator import TopAggregator

os.environ['JAVA_OPTS'] = '-Djava.security.manager=allow'

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger(__name__)
E_RMS = "e_rms"
PRINT_TIME = "time"

class PrinterAggregator(TopAggregator):
    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.model = None

        self.dataset = None

        self.default_feedrate = self.config.hyperparameters.feedrate
        self.default_acc = self.config.hyperparameters.acceleration

        self.trainer_metrics = {}
        self.trainer_settings = {}

        self.device = None
        self.test_loader = None
        
        # For tracking printer history
        self.printer_history = {}
        
        # For matrix completion
        self.spark = None
        self.als_model = None
        self.als_trained_model = None
        self.train_data = None
        self.printer_id_map = {}
        self.next_printer_id = 0
        self.condition_settings = []
        self.condition_id_map = {}
        self.new_data_points = []
        self.current_end = None
        self._rounds = 30

    def initialize(self):
        """Initialize role and matrix completion model."""
        try:
            import pyspark
            from pyspark.sql import SparkSession
            from pyspark.ml.recommendation import ALS
            from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
            
            logger.info("Initializing Spark session and ALS model")
            
            # Initialize Spark session
            self.spark = SparkSession.builder.appName("ALS 3D Printer Optimization") \
                .config("spark.memory", "4g") \
                .config("spark.sql.analyzer.failAmbiguousSelfJoin", "false") \
                .getOrCreate()
            
            # Define column names and schema
            self.COL_PRINTER = "PrinterId"
            self.COL_CONDITION = "ConditionId"
            self.COL_LOSS = "Loss"
            
            self.schema = StructType([
                StructField(self.COL_PRINTER, IntegerType()),
                StructField(self.COL_CONDITION, IntegerType()),
                StructField(self.COL_LOSS, FloatType())
            ])
            
            # Load initial data if available
            try:
                self.data_np = np.genfromtxt('large_J_mat.csv', delimiter=',') # reads the csv large_J_X_axis
                # self.data_np = np.genfromtxt('large_J_X_axis.csv', delimiter=',') # reads the csv 
                self.row_size, self.col_size = self.data_np.shape[0], self.data_np.shape[1]
                logger.info(f"Loaded initial data: {self.row_size} printers, {self.col_size} conditions") # 10 35 
                # print(f"row size: {self.row_size} \ncol size: {self.col_size}")
            except Exception as e:
                logger.error(f"Error loading initial data: {e}")
                # Create a sample dataset if file not available
                self.row_size, self.col_size = 10, 40  # Default sizes
                self.data_np = np.random.rand(self.row_size, self.col_size)
            
            # Create condition settings grid - map condition IDs to feedrate/acceleration values
            feedrate_min, feedrate_max = 50, 150  # mm/sec
            acceleration_min, acceleration_max = 4000, 7000  # mm/sec^2

            # Calculate the number of values based on the specified increments
            feedrate_increment = 25  # mm/sec
            acceleration_increment = 500  # mm/sec^2

            # Calculate how many values we'll have
            feedrate_count = int((feedrate_max - feedrate_min) / feedrate_increment) + 1  # 5 values
            acceleration_count = int((acceleration_max - acceleration_min) / acceleration_increment) + 1  # 7 values

            # Generate the values
            feedrate_values = np.linspace(feedrate_min, feedrate_max, feedrate_count)
            acceleration_values = np.linspace(acceleration_min, acceleration_max, acceleration_count)
            
            condition_id = 0
            for acc in acceleration_values:
                for feed in feedrate_values:
                    self.condition_settings.append([feed, acc])
                    self.condition_id_map[(feed, acc)] = condition_id
                    condition_id += 1
            
            # Configure ALS model for collaborative filtering
            self.als_model = ALS(
                rank=4,  # Number of latent factors
                maxIter=15,
                implicitPrefs=False,
                regParam=0.05,
                coldStartStrategy='drop',
                nonnegative=True,
                seed=42,
                userCol=self.COL_PRINTER,
                itemCol=self.COL_CONDITION,
                ratingCol=self.COL_LOSS
            )
            
            # Create initial training dataframe from loaded data
            data_train = []
            
            # Sample a subset of initial data to create training data
            # This ensures not all conditions are known initially
            known_rate = 0.45  # 45% of data is initially known
            
            for printer_idx in range(self.row_size):
                # For each printer, create a set of diagonal elements
                diag_set = [i*(self.col_size//5)+i for i in range(5)]
                diag_set.extend([5, 13])  # Add common conditions
                
                # Determine conditions to include in training
                full_set = list(range(self.col_size))
                missing_set = list(set(full_set) - set(diag_set))
                include_set = diag_set + list(np.random.choice(
                    missing_set, 
                    size=int(known_rate*self.col_size) - len(diag_set),
                    replace=False
                ))
                
                # Add selected conditions to training data
                for condition_idx in include_set:
                    data_train.append([
                        printer_idx, 
                        condition_idx, 
                        float(self.data_np[printer_idx, condition_idx])
                    ])
            
            # Convert to Spark DataFrame and train initial model
            data_train_df = pd.DataFrame(data_train, columns=[
                self.COL_PRINTER, self.COL_CONDITION, self.COL_LOSS
            ])
            self.train_data = self.spark.createDataFrame(data_train_df)
            self.als_trained_model = self.als_model.fit(self.train_data)
            
            # Create model wrapper
            class MatrixCompletionModel:
                def __init__(self, aggregator):
                    self.aggregator = aggregator
                    
                def __call__(self, err, print_time):
                    """Return optimal feedrate and acceleration based on error and print time."""
                    current_end = self.aggregator.current_end
                    
                    # Get or assign a printer ID
                    if current_end not in self.aggregator.printer_id_map:
                        self.aggregator.printer_id_map[current_end] = self.aggregator.next_printer_id
                        self.aggregator.next_printer_id += 1
                        logger.info(f"Assigned new printer ID {self.aggregator.next_printer_id-1} to {current_end}")
                    
                    printer_idx = self.aggregator.printer_id_map[current_end]
                    
                    # Calculate a combined loss that balances error and print time
                    # Lower is better for both metrics
                    normalized_err = err / 0.1  # Normalize based on expected range
                    normalized_time = print_time / 60  # Normalize to minutes
                    combined_loss = normalized_err + 0.5 * normalized_time  # Weighted sum
                    
                    # Get current condition based on current settings
                    current_feedrate = self.aggregator.trainer_settings.get(current_end, {}).get(
                        "feedrate", self.aggregator.default_feedrate)
                    current_acceleration = self.aggregator.trainer_settings.get(current_end, {}).get(
                        "acceleration", self.aggregator.default_acc)
                    
                    # Find closest condition ID from settings grid
                    closest_condition = self._find_closest_condition(current_feedrate, current_acceleration)
                    
                    # Add this data point to the collection for future training
                    self.aggregator.new_data_points.append([printer_idx, closest_condition, combined_loss])
                    
                    # Generate predictions for all conditions for this printer
                    test_conditions = []
                    for condition_idx in range(len(self.aggregator.condition_settings)):
                        test_conditions.append([printer_idx, condition_idx])
                    
                    test_df = pd.DataFrame(test_conditions, 
                                        columns=[self.aggregator.COL_PRINTER, 
                                                self.aggregator.COL_CONDITION])
                    test_spark = self.aggregator.spark.createDataFrame(test_df)
                    
                    try:
                        # Make predictions for all conditions
                        predictions = self.aggregator.als_trained_model.transform(test_spark)
                        
                        # Find the condition with minimum predicted loss
                        min_row = predictions.orderBy("prediction").first()
                        
                        if min_row:
                            best_condition_id = min_row[self.aggregator.COL_CONDITION]
                            
                            # Get the settings for this condition
                            feedrate, acceleration = self.aggregator.condition_settings[best_condition_id]
                            logger.info(f"Recommending settings for {current_end}: feedrate={feedrate}, "
                                      f"acceleration={acceleration} (condition ID {best_condition_id})")
                            return feedrate, acceleration
                        else:
                            logger.warning(f"No prediction available for {current_end}, using defaults")
                            return self.aggregator.default_feedrate, self.aggregator.default_acc
                    except Exception as e:
                        logger.error(f"Error making predictions: {e}")
                        return self.aggregator.default_feedrate, self.aggregator.default_acc
                
                def _find_closest_condition(self, feedrate, acceleration):
                    """Find the closest condition ID to the given settings."""
                    min_distance = float('inf')
                    closest_id = 0
                    
                    for idx, (cond_feedrate, cond_acc) in enumerate(self.aggregator.condition_settings):
                        # Calculate Euclidean distance with normalization
                        normalized_feedrate_diff = (feedrate - cond_feedrate) / 100
                        normalized_acc_diff = (acceleration - cond_acc) / 3000
                        distance = (normalized_feedrate_diff**2 + normalized_acc_diff**2)**0.5
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_id = idx
                    
                    return closest_id
            
            # Create model instance
            self.model = MatrixCompletionModel(self)
            
            logger.info("Matrix completion model initialized successfully")
            
        except ImportError as e:
            logger.error(f"Error importing required libraries: {e}")
            logger.warning("Falling back to default settings")
            self.model = lambda err, time: (self.default_feedrate, self.default_acc)
        except Exception as e:
            logger.error(f"Error initializing matrix completion model: {e}")
            logger.warning("Falling back to default settings")
            self.model = lambda err, time: (self.default_feedrate, self.default_acc)

    def get(self, tag: str) -> None:
        """Get data from remote role(s)."""
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        # receive local metrics from printer nodes
        for msg, metadata in channel.recv_fifo(channel.ends()):
            end, timestamp = metadata
            if not msg:
                logger.debug(f"No data from {end}; skipping it")
                continue

            logger.debug(f"Received data from {end}: {msg}")
            channel.set_end_property(end, "round_end_time", (self._round, timestamp))

            e_rms = msg[E_RMS] if E_RMS in msg else None 
            print_time = msg[PRINT_TIME] if PRINT_TIME in msg else None

            if end not in self.trainer_metrics:
                self.trainer_metrics[end] = {}
                
            self.trainer_metrics[end][E_RMS] = e_rms
            self.trainer_metrics[end][PRINT_TIME] = print_time

            # Store history for this printer
            if "history" not in self.trainer_metrics[end]:
                self.trainer_metrics[end]["history"] = []
            
            # Add current metrics to history
            self.trainer_metrics[end]["history"].append({
                E_RMS: e_rms,
                PRINT_TIME: print_time,
                "timestamp": timestamp
            })

    def put(self, tag: str) -> None:
        """Set data to remote role(s)."""
        self.dist_tag = tag
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"Channel not found for tag {tag}")
            return

        # Wait for at least one peer to join this channel
        channel.await_join()

        selected_ends = channel.ends()

        # Send out recommended settings to printer nodes
        for end in selected_ends:
            logger.debug(f"Sending settings to {end}")
            
            # Get settings for this printer
            feedrate = self.trainer_settings.get(end, {}).get("feedrate", self.default_feedrate)
            acceleration = self.trainer_settings.get(end, {}).get("acceleration", self.default_acc)
            
            channel.send(
                end,
                {
                    MessageType.ROUND: self._round,
                    "feedrate": feedrate,
                    "acceleration": acceleration,
                },
            )
            
            # Register round start time
            channel.set_end_property(end, "round_start_time", (self._round, datetime.now()))
            
            logger.info(f"Sent settings to {end}: feedrate={feedrate}, acceleration={acceleration}")

    def load_data(self) -> None:
        """Load a test dataset."""
        # Initial data is loaded in initialize()
        pass

    def train(self) -> None:
        """Update the model with new data points."""
        if not self.new_data_points or not self.spark:
            logger.debug("No new data points available for training or Spark not initialized")
            return
        
        try:
            # import pyspark
            # from pyspark.sql import SparkSession
            from pyspark.ml.recommendation import ALS
            # from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
            # Convert new data points to DataFrame
            new_data_df = pd.DataFrame(self.new_data_points, 
                                    columns=[self.COL_PRINTER, self.COL_CONDITION, self.COL_LOSS])
            
            # Convert to Spark DataFrame
            new_spark_df = self.spark.createDataFrame(new_data_df)
            
            # Update training data
            self.train_data = self.train_data.union(new_spark_df)
            
            # Retrain the model with updated settings
            self.als_model = ALS(
                rank=4,  # Latent factors
                maxIter=15,
                implicitPrefs=False,
                regParam=0.05,  # Regularization parameter
                coldStartStrategy='drop',
                nonnegative=True,
                seed=42,
                userCol=self.COL_PRINTER,
                itemCol=self.COL_CONDITION,
                ratingCol=self.COL_LOSS
            )
            
            self.als_trained_model = self.als_model.fit(self.train_data)
            
            logger.info(f"Model retrained with {len(self.new_data_points)} new data points")
            
            # Store the training history for monitoring
            if not hasattr(self, 'training_history'):
                self.training_history = []
            
            self.training_history.append({
                'timestamp': datetime.now(),
                'new_points': len(self.new_data_points),
                'total_points': self.train_data.count()
            })
            
            # Clear new data points after training
            self.new_data_points = []
            
        except Exception as e:
            logger.error(f"Error training model: {e}")

    def evaluate(self) -> None:
        """Evaluate metrics and generate optimal printer settings."""
        # Process each printer's metrics
        for end in self.trainer_metrics:
            if E_RMS not in self.trainer_metrics[end] or PRINT_TIME not in self.trainer_metrics[end]:
                logger.debug(f"No complete metrics for {end}; skipping it")
                continue
                
            # Set current printer for the model to use
            self.current_end = end
            
            # Get metrics
            err = self.trainer_metrics[end][E_RMS]
            print_time = self.trainer_metrics[end][PRINT_TIME]
            
            if err is None or print_time is None:
                logger.debug(f"Missing metrics for {end}; skipping it")
                continue
                
            # Use the model to get recommended settings
            curr_feedrate, curr_acceleration = self.model(err, print_time)
            
            # Update settings
            if end not in self.trainer_settings:
                self.trainer_settings[end] = {}
            self.trainer_settings[end]["feedrate"] = curr_feedrate
            self.trainer_settings[end]["acceleration"] = curr_acceleration
            
            # Update history with new settings
            if "history" in self.trainer_metrics[end] and self.trainer_metrics[end]["history"]:
                self.trainer_metrics[end]["history"][-1]["feedrate"] = curr_feedrate
                self.trainer_metrics[end]["history"][-1]["acceleration"] = curr_acceleration
            
            logger.info(f"Updated settings for {end}: feedrate={curr_feedrate}, acceleration={curr_acceleration}")
    
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Printer Aggregator for federated learning')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    a = PrinterAggregator(config)
    a.compose()
    a.run()
