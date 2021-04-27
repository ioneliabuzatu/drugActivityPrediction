import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy import inf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

import config


def model_RF():
    return RandomForestClassifier(n_estimators=config.n_estimators, random_state=1234, max_features=config.max_features,
                                  n_jobs=-1, class_weight=config.class_weight, max_samples=config.max_samples,
                                  min_impurity_decrease=config.min_impurity_decrease)


def model_SGD():
    return SGDClassifier(
        loss=config.log, penalty=config.penalty, max_iter=config.max_iter, learning_rate=config.learning_rate,
        eta0=config.eta0, alpha=config.alpha,
        tol=config.tol, shuffle=True, random_state=1234, class_weight=config.class_weight,
        fit_intercept=config.fit_intercept,
        n_jobs=-1)


def model_MLP():
    return MLPClassifier(hidden_layer_sizes=config.hidden_layer_sizes, activation=config.activation,
                         solver=config.solver,
                         max_iter=config.max_iter,
                         batch_size=config.batch_size,
                         learning_rate=config.learning_rate,
                         learning_rate_init=config.learning_rate_init, random_state=1234)


def model_oneVsRest():
    return OneVsRestClassifier(model_MLP())


def model_SVC():
    SVC(C=config.C, kernel=config.kernel, degree=config.degree, gamma=config.gamma, coef0=config.coef0,
        shrinking=config.shrinking,
        probability=config.probability,
        tol=config.tol, cache_size=config.cache_size, class_weight=config.class_weight, verbose=config.verbose,
        max_iter=-1,
        decision_function_shape=config.decision_function_shape, break_ties=config.break_ties, random_state=1234)


def model_DT():
    return DecisionTreeClassifier(criterion=config.criterion, splitter=config.splitter, max_depth=config.max_depth,
                                  min_samples_split=config.min_samples_split,
                                  min_samples_leaf=config.min_samples_leaf,
                                  min_weight_fraction_leaf=config.min_weight_fraction_leaf,
                                  max_features=config.max_features,
                                  random_state=config.random_state, max_leaf_nodes=config.max_leaf_nodes,
                                  min_impurity_decrease=config.min_impurity_decrease,
                                  min_impurity_split=config.min_impurity_split, class_weight=config.class_weight,
                                  ccp_alpha=config.ccp_alpha)


def model_extratree():
    return ExtraTreeClassifier(criterion=config.criterion, splitter=config.splitter, max_depth=config.max_depth,
                               min_samples_split=config.min_samples_split,
                               min_samples_leaf=config.min_samples_leaf,
                               min_weight_fraction_leaf=config.min_weight_fraction_leaf,
                               max_features=config.max_features, random_state=config.random_state,
                               max_leaf_nodes=config.max_leaf_nodes,
                               min_impurity_decrease=config.min_impurity_decrease,
                               min_impurity_split=config.min_impurity_split, class_weight=config.class_weight,
                               ccp_alpha=config.ccp_alpha)


def model_KN():
    return KNeighborsClassifier(n_neighbors=config.n_neighbors, weights=config.weights, algorithm=config.algorithm,
                                leaf_size=config.leaf_size,
                                p=config.p,
                                metric=config.metric,
                                metric_params=config.metric_params, n_jobs=config.n_jobs)


def model_perceptron():
    return Perceptron(penalty=config.penalty, alpha=config.alpha, fit_intercept=config.fit_intercept,
                      max_iter=config.max_iter,
                      tol=config.tol, shuffle=config.shuffle,
                      verbose=0, eta0=config.eta0, n_jobs=-1, random_state=1234,
                      early_stopping=config.early_stopping,
                      validation_fraction=config.validation_fraction,
                      n_iter_no_change=config.n_iter_no_change,
                      class_weight=config.class_weight,
                      warm_start=config.warm_start)


def model_LR():
    return LogisticRegression(penalty=config.penalty, dual=config.dual, tol=config.tol,
                              C=config.C, fit_intercept=config.fit_intercept,
                              intercept_scaling=config.intercept_scaling,
                              class_weight=config.class_weight, random_state=config.seed,
                              solver=config.solver, max_iter=config.max_iter,
                              multi_class=config.multi_class,
                              verbose=0, warm_start=config.warm_start,
                              n_jobs=-1, l1_ratio=config.l1_ratio)


def model_gaussian():
    return GaussianProcessClassifier(kernel=config.kernel, optimizer=config.optimizer,
                                     n_restarts_optimizer=config.n_restarts_optimizer,
                                     max_iter_predict=config.max_iter_predict,
                                     warm_start=config.warm_start, copy_X_train=config.copy_X_train,
                                     random_state=config.random_state,
                                     multi_class=config.multi_class,
                                     n_jobs=-1)


def model_GB():
    return GradientBoostingClassifier(loss=config.loss, learning_rate=config.learning_rate,
                                      n_estimators=config.n_estimators,
                                      subsample=config.subsample,
                                      criterion=config.criterion, min_samples_split=config.min_samples_split,
                                      min_samples_leaf=config.min_samples_leaf,
                                      min_weight_fraction_leaf=config.min_weight_fraction_leaf,
                                      max_depth=config.max_depth,
                                      min_impurity_decrease=config.min_impurity_decrease,
                                      min_impurity_split=config.min_impurity_split, init=config.init,
                                      random_state=config.random_state, max_features=config.max_features,
                                      verbose=0, max_leaf_nodes=config.max_leaf_nodes, warm_start=config.warm_start,
                                      validation_fraction=config.validation_fraction,
                                      n_iter_no_change=config.n_iter_no_change, tol=config.tol,
                                      ccp_alpha=config.ccp_alpha)


def model_bernoulli():
    return BernoulliNB(alpha=config.alpha, binarize=config.binarize, fit_prior=config.fit_prior,
                       class_prior=config.priors)


def model_naivegaussian():
    return GaussianNB(priors=config.priors, var_smoothing=config.var_smoothing)


def smiles_custom_dataset(filepath, test_size=0.2):
    data = pd.read_csv(filepath, index_col=0)

    data = data.replace([inf, -inf, np.nan], 0)

    x = data.drop(
        ["smiles", 'task1', "task2", "task3", "task4", "task5", "task6", "task7", "task8", "task9", "task10", "task11"],
        axis=1)
    X = np.array(x)
    y = np.array(
        data[['task1', "task2", "task3", "task4", "task5", "task6", "task7", "task8", "task9", "task10", "task11"]])

    y = y.astype('float32')
    X = X.astype("float32")

    y[y == -inf] = 0.0
    y[y == inf] = 0.0
    X[X == -inf] = 0.0
    X[X == inf] = 0.0

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    return X, y


criterion_multi_classes = [
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
]

criterion_binary_logits = [
    nn.BCEWithLogitsLoss(),
    nn.BCEWithLogitsLoss(),
    nn.BCEWithLogitsLoss(),
    nn.BCEWithLogitsLoss(),
    nn.BCEWithLogitsLoss(),
    nn.BCEWithLogitsLoss(),
    nn.BCEWithLogitsLoss(),
    nn.BCEWithLogitsLoss(),
    nn.BCEWithLogitsLoss(),
    nn.BCEWithLogitsLoss(),
    nn.BCEWithLogitsLoss(),
]

criterion_binary_loss = [
    nn.BCELoss(),
    nn.BCELoss(),
    nn.BCELoss(),
    nn.BCELoss(),
    nn.BCELoss(),
    nn.BCELoss(),
    nn.BCELoss(),
    nn.BCELoss(),
    nn.BCELoss(),
    nn.BCELoss(),
    nn.BCELoss(),
]

models = {
    "model_RF": model_RF(), "model_SGD": model_SGD(), "model_MLP": model_MLP(), "model_oneVsRest": model_oneVsRest(),
    "model_SVC": model_SVC(), "model_DT": model_DT(), "model_extratree": model_extratree(), "model_KN": model_KN(),
    "model_perceptron": model_perceptron(), "model_LR": model_LR(), "model_gaussian": model_gaussian(),
    "model_GB": model_GB(), "model_bernoulli": model_bernoulli(), "model_naivegaussian": model_naivegaussian()
}
