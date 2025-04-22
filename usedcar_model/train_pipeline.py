# pylint: disable-all

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from usedcar_model.config.core import config
from usedcar_model.pipeline import xgb_model
from usedcar_model.processing.data_manager import load_dataset, save_pipeline

from clearml import Task

# Initialize ClearML Task
task = Task.init(project_name="Vehicle Price Prediction", task_name="XGBoost Model") 

def run_training() -> None:
    
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name = config.app_config_.training_data_file)
    
    # Log dataset information
    task.upload_artifact('dataset', artifact_object=data.describe())

    
    print("Training data shape:", data.shape)
    
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        
        data[config.model_config_.features],     # predictors
        data[config.model_config_.target],       # target
        test_size = config.model_config_.test_size,
        random_state=config.model_config_.random_state,   # set the random seed here for reproducibility
    )
    
    # Log dataset split sizes
    task.logger.report_scalar(
        "dataset_sizes", "train_samples", value=len(X_train), iteration=0
    )
    task.logger.report_scalar(
        "dataset_sizes", "test_samples", value=len(X_test), iteration=0
    )

    # Pipeline fitting
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    
    # Connect hyperparameters to ClearML for tracking
    task.connect(xgb_model.named_steps['regressor'].get_params())  # This allows you to modify parameters from the UI

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate the score/error
    print(f"Mean Absolute Error: ${mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Log metrics to ClearML
    task.logger.report_scalar("performance", "MAE", value=mae, iteration=0)
    task.logger.report_scalar("performance", "R2", value=r2, iteration=0)

    # persist trained model
    model_file_name = save_pipeline(pipeline_to_persist = xgb_model)
    
     # Register the model in ClearML
    task.upload_artifact('model', artifact_object=model_file_name)
    
if __name__ == "__main__":
    run_training()