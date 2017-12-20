import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.optimizers import SGD, Adam, Adagrad
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
    x_tr, x_v, y_tr, y_v = train_test_split(x_tr, y_tr, test_size=val_size)
    return(x_tr,x_v,x_t,y_tr,y_v,y_t)

def create_model_3(	filters1=32,
                    filters2=32,
					kernel1=(3,3),
                    kernel2=(3,3),
                    pool=(2,2),
                    units=128,
					optimizer="adam", 
					lr=0.01, 
					dropout=0):

    model = Sequential()
    model.add(Conv2D(filters1, kernel1, activation='relu', input_shape=(28,28,1)))
    model.add(BatchNormalization(axis=-1, center=False))
    model.add(Conv2D(filters2, kernel2, activation='relu'))
    model.add(BatchNormalization(axis=-1, center=False))
    model.add(MaxPooling2D(pool_size=pool))
    model.add(Flatten())
    model.add(Dense(units, activation='relu'))
    model.add(BatchNormalization(center=False))
    model.add(Dropout(dropout))
    model.add(Dense(25, activation='softmax'))

    if optimizer=="adam":
        opt = Adam(lr=lr)
    else:
    	opt = Adagrad(lr=lr)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

with open("MM Model 3", "w") as file:
	print("Hyperparameter validation for MinMax scaled data\n", file=file)

def write_results(best_params, best_score):
	with open("MM Model 3", "a") as file:
		for (k,v) in best_params.items():
		    if k=="lam" and best_params["regularizer"]==None:
		        continue
		    print("{:<20}{}".format(k, v), file=file)
		print("{:<20}{}".format("MODEL 3 BEST SCORE ", best_score), file=file)

x_tr, x_v, x_t, y_tr, y_v , y_t = load_data()

mm_scaler = MinMaxScaler()
x_tr_mm = mm_scaler.fit_transform(x_tr)
x_v_mm = mm_scaler.transform(x_v)
x_t_mm = mm_scaler.transform(x_t)

x_v_mm = x_v_mm.reshape(x_v_mm.shape[0], 28, 28, 1)

clf = KerasClassifier(build_fn=create_model_3)
units = [32,64,128]
optimizers = ["adam", "adagrad"]
filters = [16,32]
kernels = [(3,3), (5,5), (7,7)]
pools = [(2,2)]
dropouts = [0.2, 0.5]
epochs = [10,30]
batch_size = [64]
lrs = [0.1,0.01,0.001]

param_grid = dict(	optimizer=optimizers, 
					units=units,
                    pool=pools,
					filters1=filters,
                    filters2=filters,
					lr=lrs,
					dropout = dropouts,
					kernel1 = kernels,
                    kernel2 = kernels,
					epochs=epochs, 
					batch_size=batch_size)

grid = GridSearchCV(estimator=clf, param_grid=param_grid)
grid_result = grid.fit(x_v_mm, to_categorical(y_v))
write_results(grid_result.best_params_, grid_result.best_score_)