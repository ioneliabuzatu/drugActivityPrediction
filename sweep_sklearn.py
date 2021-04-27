import warnings

import numpy as np
import pandas as pd
import wandb
from numpy import inf
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

import config
from utils import models

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

wandb.run = config.tensorboard.run


def load_and_clean_train_csv(filepath, test_size=0.2):
    data = pd.read_csv(filepath, index_col=0)

    data = data.replace([np.inf, -np.inf, np.nan], 0)

    x = data.drop(
        ["smiles", 'task1', "task2", "task3", "task4", "task5", "task6", "task7", "task8", "task9", "task10", "task11"],
        axis=1)
    X = np.array(x)
    y = np.array(
        data[['task1', "task2", "task3", "task4", "task5", "task6", "task7", "task8", "task9", "task10", "task11"]])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    y_train = y_train.astype('float32')
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    y_train[y_train == -inf] = 0.0
    y_train[y_train == inf] = 0.0
    X_train[X_train == -inf] = 0.0
    X_train[X_train == inf] = 0.0
    y_test[y_test == inf] = 0.0
    y_test[y_test == -inf] = 0.0
    X_test[X_test == inf] = 0.0
    X_test[X_test == -inf] = 0.0

    return X_train, X_test, y_train, y_test


def model_fit_and_predict(model, x, y, idx, task, test):
    model.fit(x[idx], y[idx, task])
    prediction = model.predict(test)
    return prediction


X_train, X_test, y_train, y_test = load_and_clean_train_csv("./data/data_train_descriptors.csv")

for task in range(11):
    idx_task = (y_train[:, task] != 0)

    prediction = model_fit_and_predict(models[config.model_name], X_train, y_train, idx_task, task, X_test)

    accuracy = np.mean(y_test[:, task] == prediction)

    wandb.log({f"Tasks_accuracy": accuracy}, step=task)
