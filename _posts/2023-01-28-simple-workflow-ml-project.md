---
layout: post
title: A (very) simple ML workflow for beginners
date: 2023-02-08 11:25:00
description:
tags: fundamentals inanutshell workflow
categories: fundamentals-engineering
thumbnail: assets/img/simple_mlworkflow/thumbnail.jpg
giscus_comments: true
related_posts: true
toc:
  sidebar: right
---

# Simple workflow of ML problem
Below is a very simple workflow in a ML project

*(Please note that in practices, for larger ML project, things are very complicated, not just simple like this!!!)*

<!-- ![overall_ml_process](/img/exp2eng/overall_ml_process.png) -->
{% include figure.html path="/assets/img/simple_mlworkflow/overall_ml_process.png" class="img-fluid rounded z-depth-1" %}

Each step will include serveral "actions", for example

* **Preprocess**: fill NaN data, remove noise data
* **Feature Engineering**: encode text data to numerical data, scale data
* **Modelling**: set up hyper-parameters, build model
* **Training**: split data, train model
* **Predict**: predict the outcome

Each "action" is generalized as a **function**. As a result, let's say, for each step, we will have corresponding functions

* **Preprocess**: `fill_na`, `remove_noise`
* **Feature Engineering**: `encode_data`, `scale_data`
* **Modelling**: `setup_params`, `build_model`
* **Training**: `split_data`, `train_model`
* **Predict**: `predict`

# From experiment in notebook

In Google Colab (or, JupyterLab), this is something as below codes

```python
# preprocess

def remove_noise(data):
  pass

def fill_na(data):
  pass
```

```python
# feature engineering

def encode_data(data):
  pass

def scale_data(data):
  pass
```

```python
# modelling
def setup_params():
  pass

def build_model(params):
  pass

```

```python
# training

def split_data(data):
  pass


def train_model(model, train_data):
  pass
```

```python
# predict

def predict(model, data_to_predict):
  pass
```

Then, you may call above functions to run the experiments

```python
# import libraries
import pandas as pd
```

```python
# load data
data = pd.read_csv('dog_vs_cat.csv')
```

```python
# preprocess
data = remove_noise(data)
data = fill_na(data)

# feature engineering
data = encode_data(data)
data = scale_data(data)

# split data
X_train, y_train, X_test, y_test = split_data(data)
```

```python
# train model

# set up
params = setup_params()
model = build_model(params)

# train
model = train_model(model=model, train_data=(X_train, y_train))
```

After training, we may use the trained_model to predict an outcome

First, we want to test on the "test_data" that we have split

```python
# validate the test data
test_outcomes = predict(model, data_to_predict=(X_test, y_test))
```

Then, we will you data comming from real world to see how the model reacts
Note that, this data comming from real world **HAVE NOT BEEN PROCESSED yet**. So we need to use functions in "preprocess" step

```python
real_data = pd.read_csv('real_dogcat_data.csv')

# We have to PREPROCESS this data (as what we have done with training data)
# preprocess
real_data = remove_noise(real_data)
real_data = fill_na(real_data)

# feature engineering
real_data = encode_data(real_data)
real_data = scale_data(real_data)

# (this is real data, we do not need to split them)
```

Finally, we use `predict` function to see the outcomes

```python
outcomes = predict(model, real_data)
```

# To engineering in VSCode

The above steps you may run serveral times, and "tune" to have a "good enough" model. After that, you may wish to **"bring" your model into an application**

To do this, we have to "engineering" the codes above

**"Engineering"** process is to arrange the above code in structure. A structure is something like a folder structure in your computer. Technically, we will transform ".ipynb" to ".py" file

What to arrange? We arrange the functions

How to arrange? There are many ways
* the most basic one is to leverage the above ML process, functions in the same steps will be in the same folder
* another way is to arrange by their functionalities


In the general process, we have these 2 steps:
* **Preprocess**: `fill_na`, `remove_noise`
* **Feature Engineering**: `encode_data`, `scale_data`

These functions generally handle data, so we can group them into something called **processing**

Next, we have this step
* **Modelling**: `setup_params`, `build_model`
These functions are to build the model, so we can group them into **models**

Finally, we have 2 final steps:
* **Training**: `split_data`, `train_model`
* **Predict**: `predict`

These functions are to support the process of training, testing, spliting data, predicting, so we can group them into **utils**


As a result, our "structure" will be
* **processing**: includes preprocess, and feature engineering
* **models**: setup params, build model
* **utils**: training and predict

Each "structure" will have "steps", these "steps" can be generalized as ".py" files.
Each "step" have functions, these functions will be written in the corresponding ".py" files

Below is the resulted structure
<!-- ![ml_sample_structure](/img/exp2eng/ml_sample_structure.png) -->
{% include figure.html path="/assets/img/simple_mlworkflow/ml_sample_structure.png" class="img-fluid rounded z-depth-1" %}


In the folder **models**, we can be more specific to create two `cls_models.py` and `reg_models.py` representing "classification models" and "regression models". (It's up to you)

In the root, we have `main.py` as the "entry" file. "Entry" file is something containing "main code" to run. Something like this (the cell in the notebook)

<!-- ![main](/img/exp2eng/main.png) -->
{% include figure.html path="/assets/img/simple_mlworkflow/main.png" class="img-fluid rounded z-depth-1" %}


Here, we start to "bring" the code in the notebook to the corresponding "folder" in the "structure". As a result, we will have

In the `preprocessing` > `feature_engineering.py`

<!-- ![feature_engineering](/img/exp2eng/feature_engineering.png) -->
{% include figure.html path="/assets/img/simple_mlworkflow/feature_engineering.png" class="img-fluid rounded z-depth-1" %}


In the `processing` > `preprocess.py`

<!-- ![preprocess](/img/exp2eng/preprocess.png) -->
{% include figure.html path="/assets/img/simple_mlworkflow/preprocess.png" class="img-fluid rounded z-depth-1" %}



In the `models` > `params.py`

<!-- ![params](/img/exp2eng/params.png) -->
{% include figure.html path="/assets/img/simple_mlworkflow/params.png" class="img-fluid rounded z-depth-1" %}


In the `models` > `cls_models.py`

<!-- ![cls_models](/img/exp2eng/cls_models.png) -->
{% include figure.html path="/assets/img/simple_mlworkflow/cls_models.png" class="img-fluid rounded z-depth-1" %}


In the `utils` > `split_data.py`

<!-- ![split_data](/img/exp2eng/split_data.png) -->
{% include figure.html path="/assets/img/simple_mlworkflow/split_data.png" class="img-fluid rounded z-depth-1" %}


In the `utils` > `training.py`

<!-- ![training](/img/exp2eng/training.png) -->
{% include figure.html path="/assets/img/simple_mlworkflow/training.png" class="img-fluid rounded z-depth-1" %}


In the `utils` > `predicting.py`

<!-- ![predicting](/img/exp2eng/predicting.png) -->
{% include figure.html path="/assets/img/simple_mlworkflow/predicting.png" class="img-fluid rounded z-depth-1" %}


In the `main.py`, we have

<!-- ![final_main](/img/exp2eng/final_main.png) -->
{% include figure.html path="/assets/img/simple_mlworkflow/final_main.png" class="img-fluid rounded z-depth-1" %}


Different to notebook which all functions are in the same notebook, functions in engineering are seperated in many ".py" files, so we need to **import** them

<!-- ![import](/img/exp2eng/import.png) -->
{% include figure.html path="/assets/img/simple_mlworkflow/import.png" class="img-fluid rounded z-depth-1" %}

# Wrap up
A simple workflow of ML process
<!-- ![wrapup_ml_process](/img/exp2eng/overall_ml_process.png) -->
{% include figure.html path="/assets/img/simple_mlworkflow/overall_ml_process.png" class="img-fluid rounded z-depth-1" %}

Experiment in notebook
* Each "step" includes many "actions"
* These "actions" are generalized as "functions"
* In notebook, all functions are in the same notebook (.ipynb files)

Engineering in VSCode
* Convert notebook ".ipynb" into many ".py" files in a structure
* A structure has "folders" are group of "steps"
* Each "step" is a ".py" files
* Each ".py" file contains many "functions"
* Bring the "functions" in notebook into corresponding ".py" files
* A structure has "main.py" file as the "entry" file
* In the entry file, we can not directly call the functions, we need to "import" them from ".py" files in the structure


# Demo
The repo can be found [here](https://github.com/tranminhquan/simple-ml-template)