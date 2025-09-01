import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
from mlProject.utils.common import save_json
import numpy as np
import joblib 
from mlProject.entity.config_entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig) -> None:
        self.config = config
        
    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_squared_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def save_results(self):
        # Load the test dataset
        test_data = pd.read_csv(self.config.test_data_path)

        # Load the trained model
        model = joblib.load(self.config.model_path)

        # Split features (X) and target (y)
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]

        # Make predictions
        predicted_qualities = model.predict(test_x)

        # Evaluate predictions
        (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)

        # Saving metrics as local JSON file
        scores = {"rmse": rmse, "mae": mae, "r2": r2}
        save_json(path=Path(self.config.metric_file_name), data=scores)