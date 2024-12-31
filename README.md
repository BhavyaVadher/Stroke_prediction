# End-to-end-Machine-Learning-Project-with-MLflow

## Preprocessing Steps for EDA

	1. Missing Values
	2. Duplicate Values
	3. Removing outliers
	4. Feature Selection
	5. Scaling of data
	6. Encoding of features
	7. Feature Engineering
	8. Dimensionality reduction
	
## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the app.py



# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/Ashokmevada/Stroke_prediction.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -p stroke python -y
```

```bash
conda activate stroke
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```



## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/ashokmevada18/Stroke_prediction.mlflow 
MLFLOW_TRACKING_USERNAME=ashokmevada18 
MLFLOW_TRACKING_PASSWORD=ffbbf19eb49ca4a5b91f9d1001c49b1c310a8cd0 
python script.py

Run this to export as env variables:


```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/ashokmevada18/Stroke_prediction.mlflow

export MLFLOW_TRACKING_USERNAME=ashokmevada18 

export MLFLOW_TRACKING_PASSWORD=ffbbf19eb49ca4a5b91f9d1001c49b1c310a8cd0

```

MLFLOW_TRACKING_URI=https://dagshub.com/ashokmevada18/Stroke_prediction.mlflow \
MLFLOW_TRACKING_USERNAME=ashokmevada18 \
MLFLOW_TRACKING_PASSWORD=ffbbf19eb49ca4a5b91f9d1001c49b1c310a8cd0 \
python script.py





## About MLflow 
MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & tagging your model


