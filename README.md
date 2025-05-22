# KidneyDiseaseClassification
Deep Learning and computer vision


## Workflows

1.  Update config.yaml
2.  Update secrets.yaml [Optional]
3.  Update params.yaml
4.  Update the entity
5.  Update the configuration manager in src config
6.  Update the components
7.  Update the pipeline
8.  Update the main.py
9.  Update the dvc.yaml
10. app.py

# How to run?

### STEPS:

Clone the Repository

```bash
https://github.com/krishnaik06/kidney-Disease-Classification-Deep-Learning-Project
```

### STEP 01 - Create a conda environment after cloning the repository

```bash
conda create -n cnncls python=3.8 -y
```

```bash
conda activate cnncls
```

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI = https://dagshub.com/Vyshnavi-d-p-3/KidneyDiseaseClassification.mlflow
MLFLOW_TRACKING_USERNAME = Vyshnavi-d-p-3
MLFLOW_TRACKING_PASSWORD = 24ddf9e30772a735f3355accd6c53766fca99e41

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/Vyshnavi-d-p-3/KidneyDiseaseClassification.mlflow

export MLFOW_TRACKING_USERNAME=Vyshnavi-d-p-3

export MLFLOW_TRACKING_PASSWORD=24ddf9e30772a735f3355accd6c53766fca99e41

```

### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag


### Documentation Frive link
[Documentation](https://drive.google.com/file/d/12s8I8dgiHOsQylGwsoVcNJX2UXHsfezD/view?usp=sharing)
