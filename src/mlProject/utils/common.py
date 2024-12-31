import os
from box.exceptions import BoxValueError
import yaml
from src.mlProject import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import sys
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from urllib.parse import urlparse



# Suppress all warnings
warnings.filterwarnings("ignore")



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data



@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def evaluteModel(X_train, X_test, y_train, y_test, models, params , model_evaluation_config):
    
    try:       

        mlflow.set_registry_uri(model_evaluation_config.mlflow_uri)
        print(model_evaluation_config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        report = {}

        # Start MLflow run
        with mlflow.start_run(run_name= "Model Training"):

            for model_name, model in models.items():

                # Start nested MLflow run
                with mlflow.start_run(nested=True , run_name = model_name):
                    parameters = params[model_name]

                    # Perform GridSearchCV
                    gs = GridSearchCV(model, param_grid=parameters, cv=2)
                    gs.fit(X_train, y_train)

                    with mlflow.start_run(nested=True):
                        mlflow.set_experiment("final Parameter Run")

                        for i, params_ in enumerate(gs.cv_results_['params']):
                            with mlflow.start_run(nested=True):
                                #Log individual parameters
                                mlflow.set_experiment("parameters Run")
                                mlflow.log_params(params_)
                                mlflow.log_param(model_name , "algorithm")
                                mlflow.log_metric('mean_test_score', gs.cv_results_['mean_test_score'][i])

                        # Set model parameters to best parameters found by GridSearchCV
                        model.set_params(**gs.best_params_)
                        model.fit(X_train, y_train)

                        # Make predictions
                        y_train_pred = model.predict(X_train)
                        y_test_pred = model.predict(X_test)

                        # Calculate evaluation metrics
                        train_accuracy = accuracy_score(y_train_pred, y_train)
                        test_accuracy = accuracy_score(y_test_pred, y_test)
                        precision = precision_score(y_test, y_test_pred)
                        recall = recall_score(y_test, y_test_pred)
                        f1 = f1_score(y_test, y_test_pred)


                        scores = {"train_accuracy": train_accuracy, "test_accuracy": test_accuracy, "precision": precision, "recall": recall, "f1": f1}
                        path = Path(model_evaluation_config.metric_file_name) / (model_name + ".json")
                        save_json(path = path , data = scores)                      

                        #Log model final metrics to MLflow
                        # mlflow.log_model(model_name = model_name , model = model)
                        mlflow.log_params(gs.best_params_)
                        mlflow.log_metric("train_accuracy", train_accuracy)
                        mlflow.log_metric("test_accuracy", test_accuracy)
                        mlflow.log_metric("precision", precision)
                        mlflow.log_metric("recall", recall)
                        mlflow.log_metric("f1_score", f1)  

                    # Add model score to report              
                    report[model_name] = test_accuracy

                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
                    print("Model registered in DAGSHUB")
                else:
                    mlflow.sklearn.log_model(model, "model")
                    print("Model saved locally")

        return report

    except Exception as e:
        
        raise Exception(e , sys)

