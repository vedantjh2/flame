# printer_matrix_completion.py
import numpy as np
import pandas as pd
import os
import logging
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
import pyspark.sql.functions as F

logger = logging.getLogger(__name__)

class PrinterMatrixCompletion:
    """3D Printer optimization using matrix completion with ALS"""
    
    def __init__(self, config=None):
        """Initialize the matrix completion model for printer optimization"""
        self.config = config
        self.spark = None
        self.model = None
        self.train_data = None
        self.trained = False
        self.condition_map = None  # Maps condition index to (feedrate, acceleration)
        self.matrix_X = None  # Printer factors
        self.matrix_Y = None  # Condition factors
        
        # Define schema for data
        self.schema = StructType([
            StructField("PrinterId", IntegerType()),
            StructField("ConditionId", IntegerType()),
            StructField("Loss", FloatType())
        ])
        
        # ALS model configuration
        self.als_config = {
            "rank": 4,  # Rank for collaborative approach
            "maxIter": 15,
            "implicitPrefs": False,
            "regParam": 0.05,
            "coldStartStrategy": 'drop',
            "nonnegative": True,
            "seed": 42,
            "userCol": "PrinterId",
            "itemCol": "ConditionId",
            "ratingCol": "Loss"
        }
    
    def _init_spark(self):
        """Initialize Spark session"""
        try:
            self.spark = SparkSession.builder.appName("ALS 3D Printer Optimization") \
                .config("spark.memory", "4g") \
                .config("spark.sql.analyzer.failAmbiguousSelfJoin", "false") \
                .getOrCreate()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Spark: {e}")
            return False
    
    def initialize(self, training_data_path=None, condition_map_path=None):
        """Initialize the model with training data"""
        # Initialize Spark if needed
        if not self.spark and not self._init_spark():
            return False
            
        # Load training data
        if training_data_path and os.path.exists(training_data_path):
            try:
                self.train_data = self.spark.read.csv(path=training_data_path, schema=self.schema)
                logger.info(f"Loaded {self.train_data.count()} training records")
            except Exception as e:
                logger.error(f"Failed to load training data: {e}")
                self.train_data = self.spark.createDataFrame([], self.schema)
        else:
            # Create empty DataFrame
            self.train_data = self.spark.createDataFrame([], self.schema)
        
        # Load condition map
        if condition_map_path and os.path.exists(condition_map_path):
            try:
                self.condition_map = pd.read_csv(condition_map_path)
                logger.info(f"Loaded {len(self.condition_map)} condition mappings")
            except Exception as e:
                logger.error(f"Failed to load condition map: {e}")
        
        # Initialize ALS model
        try:
            self.als = ALS(**self.als_config)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ALS model: {e}")
            return False
    
    def train(self):
        """Train the ALS model with available data"""
        if self.train_data is None or self.train_data.count() < 2:
            logger.warning("Not enough training data (need at least 2 records)")
            return False
            
        try:
            # Train model
            self.model = self.als.fit(self.train_data)
            
            # Extract latent factors
            self.user_factors = self.model.userFactors
            self.item_factors = self.model.itemFactors
            
            # Extract matrices for faster predictions
            self._extract_factor_matrices()
            
            self.trained = True
            logger.info("Model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            return False
    
    def _extract_factor_matrices(self):
        """Extract latent factor matrices from model"""
        try:
            # Convert to numpy arrays
            user_factors_np = np.array(self.user_factors.select("features").rdd.map(lambda row: row[0]).collect())
            item_factors_np = np.array(self.item_factors.select("features").rdd.map(lambda row: row[0]).collect())
            id_index_user = np.array(self.user_factors.select("id").rdd.map(lambda row: row[0]).collect())
            id_index_item = np.array(self.item_factors.select("id").rdd.map(lambda row: row[0]).collect())
            
            # Create matrices from factors
            max_user_id = int(max(id_index_user)) + 1
            max_item_id = int(max(id_index_item)) + 1
            
            self.matrix_X = np.zeros((max_user_id, user_factors_np.shape[1]))
            for j in range(user_factors_np.shape[0]):
                self.matrix_X[int(id_index_user[j]), :] = user_factors_np[j, :]
            
            self.matrix_Y = np.zeros((max_item_id, item_factors_np.shape[1]))
            for j in range(item_factors_np.shape[0]):
                self.matrix_Y[int(id_index_item[j]), :] = item_factors_np[j, :]
                
        except Exception as e:
            logger.error(f"Failed to extract factor matrices: {e}")
    
    def add_observation(self, printer_id, condition_id, loss):
        """Add a new observation to the training data"""
        from pyspark.sql import Row
        
        try:
            # Create new row
            new_row = self.spark.createDataFrame([
                Row(PrinterId=int(printer_id), ConditionId=int(condition_id), Loss=float(loss))
            ])
            
            # Add to training data
            if self.train_data is None:
                self.train_data = new_row
            else:
                self.train_data = self.train_data.union(new_row)
            
            # Mark model as needing retraining
            self.trained = False
            return True
            
        except Exception as e:
            logger.error(f"Failed to add observation: {e}")
            return False
    
    def predict_best_condition(self, printer_id):
        """Predict the best printing condition for a printer
        
        Args:
            printer_id: ID of the printer
            
        Returns:
            condition_id, (feedrate, acceleration)
        """
        if not self.trained:
            success = self.train()
            if not success:
                return self._get_default_settings()
        
        try:
            printer_id = int(printer_id)
            
            # Get all condition IDs
            available_conditions = list(range(len(self.matrix_Y)))
            
            # Calculate predicted loss for each condition
            utility = []
            for condition_id in available_conditions:
                if printer_id < self.matrix_X.shape[0] and condition_id < self.matrix_Y.shape[0]:
                    # Calculate predicted loss (smaller is better)
                    predicted_loss = np.dot(self.matrix_X[printer_id], self.matrix_Y[condition_id])
                    utility.append((condition_id, predicted_loss))
            
            if not utility:
                return self._get_default_settings()
                
            # Sort by predicted loss (ascending - smaller is better)
            utility.sort(key=lambda x: x[1])
            
            # Get the best condition ID
            best_condition_id = utility[0][0]
            
            # Map to feedrate and acceleration
            return best_condition_id, self.lookup_condition(best_condition_id)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._get_default_settings()
    
    def create_condition_map(self, feedrates, accelerations):
        """Create a mapping from condition ID to (feedrate, acceleration)"""
        import itertools
        
        try:
            # Create all combinations of feedrate and acceleration
            combinations = list(itertools.product(feedrates, accelerations))
            
            # Create DataFrame with condition ID mapping
            data = {
                'condition_id': range(len(combinations)),
                'feedrate': [c[0] for c in combinations],
                'acceleration': [c[1] for c in combinations]
            }
            
            self.condition_map = pd.DataFrame(data)
            return self.condition_map
            
        except Exception as e:
            logger.error(f"Failed to create condition map: {e}")
            return pd.DataFrame(columns=['condition_id', 'feedrate', 'acceleration'])
    
    def lookup_condition(self, condition_id):
        """Get feedrate and acceleration for a condition ID"""
        try:
            if self.condition_map is None:
                return self._get_default_values()
                
            row = self.condition_map[self.condition_map['condition_id'] == condition_id]
            if row.empty:
                return self._get_default_values()
                
            return row['feedrate'].values[0], row['acceleration'].values[0]
            
        except Exception as e:
            logger.error(f"Failed to lookup condition: {e}")
            return self._get_default_values()
    
    def get_condition_id(self, feedrate, acceleration):
        """Get condition ID for feedrate and acceleration settings"""
        try:
            if self.condition_map is None:
                return 0
                
            # Find matching row
            row = self.condition_map[
                (self.condition_map['feedrate'] == feedrate) & 
                (self.condition_map['acceleration'] == acceleration)
            ]
            
            if row.empty:
                return 0
                
            return row['condition_id'].values[0]
            
        except Exception as e:
            logger.error(f"Failed to get condition ID: {e}")
            return 0
    
    def _get_default_values(self):
        """Get default feedrate and acceleration values"""
        default_feedrate = 100
        default_acc = 8000
        if self.config and hasattr(self.config, 'hyperparameters'):
            default_feedrate = getattr(self.config.hyperparameters, 'feedrate', default_feedrate)
            default_acc = getattr(self.config.hyperparameters, 'acceleration', default_acc)
        return default_feedrate, default_acc
    
    def _get_default_settings(self):
        """Get default condition ID and settings"""
        return 0, self._get_default_values()
    
    def save_data(self, training_path="data/training_data.csv"):
        """Save training data to disk"""
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(training_path), exist_ok=True)
            
            # Save training data
            if self.train_data is not None:
                train_pd = self.train_data.toPandas()
                train_pd.to_csv(training_path, index=False)
                logger.info(f"Saved {len(train_pd)} training records")
            
            return True
                
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            return False
    
    def shutdown(self):
        """Clean up resources"""
        if self.spark:
            self.spark.stop()
            self.spark = None
