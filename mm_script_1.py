import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.regularizers import l1,l2
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

np.random.seed(42)

def load_data(val_size=5000):
    train = pd.read_csv("sign_mnist_train.csv")
    test = pd.read_csv("sign_mnist_test.csv")
    y_tr = train["label"]
    x_tr = train.iloc[:,1:]
    y_t = test["label"]
    x_t = test.iloc[:,1:]
    i = np.random.randint(y_tr.shape[0]+1, size=val_size)
    y_v = y_tr.take(i) 
    x_v = x_tr.take(i)
    return(x_tr,x_v,x_t,y_tr,y_v,y_t)

def create_model_1(	units=1, 
					activation="relu",
					optimizer="sgd", 
					lr=0.01, 
					dropout=0,):

    model = Sequential()
    model.add(Dense(units, input_dim=784, kernel_initializer="uniform", activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(25, kernel_initializer="uniform", activation="softmax"))
    if optimizer=="sgd":
        opt = SGD(lr=lr)
    elif optimizer=="rmsprop":
        opt = RMSprop(lr=lr)
    else:
    	opt = Adagrad(lr=lr)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

with open("MM Model 1", "w") as file:
	print("Hyperparameter validation for MinMax scaled data\n", file=file)

def write_results(best_params, best_score):
	with open("MM Model 1", "a") as file:
		for (k,v) in best_params.items():
		    if k=="lam" and best_params["regularizer"]==None:
		        continue
		    print("{:<20}{}".format(k, v), file=file)
		print("{:<20}{}".format("MODEL 1 BEST SCORE ", best_score), file=file)

x_tr, x_v, x_t, y_tr, y_v , y_t = load_data()

mm_scaler = MinMaxScaler()
x_tr_mm = mm_scaler.fit_transform(x_tr)
x_v_mm = mm_scaler.transform(x_v)
x_t_mm = mm_scaler.transform(x_t)

clf = KerasClassifier(build_fn=create_model_1)
units = [1,25,50,100]
activations = ["sigmoid", "relu"]
optimizers = ["sgd", "rmsprop", "adagrad"]
lrs = [0.1,0.01,0.001]
dropouts = [0, 0.2, 0.5]
epochs = [50,150]
batch_size = [64,128]
param_grid = dict(	optimizer=optimizers, 
					units=units, 
					activation=activations,
					lr=lrs,
					dropout = dropouts,
					epochs=epochs, 
					batch_size=batch_size)

grid = GridSearchCV(estimator=clf, param_grid=param_grid)
grid_result = grid.fit(x_v_mm, to_categorical(y_v))
write_results(grid_result.best_params_, grid_result.best_score_)