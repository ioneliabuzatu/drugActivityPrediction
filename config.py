import experiment_buddy

seed = 123
num_classes = 3
model_name = "model_SGD"
filepath_data_train = "../data/data_train_descriptors.csv"
# filepath_data_train = "/home/mila/g/golemofl/data/data_train_descriptors.csv"
n_estimators = 200
criterion = "entropy"
max_features = "auto"
bootstrap = True
class_weight = "balanced"
max_samples = 100
min_impurity_decrease = 0.0

loss = "log"
penalty = "l2"
max_iter = 1000
learning_rate = "adaptive"
eta0 = 1
alpha = 0.1
tol = 0.001
shuffle = True
random_state = 1234
fit_intercept = True

hidden_layer_sizes = 3000
activation = "relu"
solver = "adam"
batch_size = 64
learning_rate_init = 0.001

C = 1.0
kernel = 'rbf'
degree = 3
gamma = 'scale'
coef0 = 0.0
shrinking = True
probability = True
cache_size = 200
decision_function_shape = 'ovr'
break_ties = True

splitter = 'best'
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
min_weight_fraction_leaf = 0.0
max_leaf_nodes = None
min_impurity_split = None
ccp_alpha = 0.0

n_neighbors = 5
weights = 'uniform'
algorithm = 'auto'
leaf_size = 30
p = 2
metric = 'minkowski'
metric_params = None

verbose = 0
n_jobs = None
early_stopping = False
validation_fraction = 0.1
n_iter_no_change = 5
warm_start = False

dual = False
intercept_scaling = 1
multi_class = 'auto'
l1_ratio = None

optimizer = 'fmin_l_bfgs_b'
n_restarts_optimizer = 0
max_iter_predict = 100
copy_X_train = True

subsample = 1.0
init = None

binarize = 0.0
fit_prior = True
class_prior = None

priors = None
var_smoothing = 1e-09

experiment_buddy.register(locals())
tensorboard = experiment_buddy.deploy(
    "mila",
    sweep_yaml="./sweep.yaml",
    proc_num=99,
    wandb_kwargs={"entity": "ionelia"}
)
