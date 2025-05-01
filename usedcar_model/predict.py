# pylint: disable-all
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from usedcar_model import __version__ as _version
from usedcar_model.config.core import config
from usedcar_model.processing.data_manager import load_pipeline
from usedcar_model.processing.data_manager import pre_pipeline_preparation
from usedcar_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
usedcar_pipe = load_pipeline(file_name = pipeline_file_name)

def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    
    validated_data, errors = validate_inputs(input_df = pd.DataFrame(input_data))
    
    #validated_data = validated_data.reindex(columns = ['dteday', 'season', 'hr', 'holiday', 'weekday', 'workingday', 
    #                                                   'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'yr', 'mnth'])
    validated_data = validated_data.reindex(columns = config.model_config_.features)
    
    results = {"predictions": None, "version": _version, "errors": errors}
      
    if not errors:
        predictions = usedcar_pipe.predict(validated_data)
        results = {"predictions": np.floor(predictions), "version": _version, "errors": errors}
        print(results)

    return results

if __name__ == "__main__":
    data_in = {
            'age': [65.0],
            'anaemia': [1],
            'creatinine_phosphokinase': [160],
            'diabetes': [1],
            'ejection_fraction': [20],
            'high_blood_pressure': [0],
            'platelets': [327000.00],
            'serum_creatinine': [27.00],
            'serum_sodium': [116],
            'sex': [0],
            'smoking': [0],
            'time': [8]
    }
    
    make_prediction(input_data = data_in)