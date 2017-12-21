import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

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

with open("No PP SVM", "w") as file:
	print("Hyperparameter validation for MinMax scaled data\n", file=file)

def write_results(best_params, best_score):
	with open("No PP SVM", "a") as file:
		for (k,v) in best_params.items():
		    print("{:<20}{}".format(k, v), file=file)
		print("{:<20}{}".format("SVM BEST SCORE ", best_score), file=file)

x_tr, x_v, x_t, y_tr, y_v , y_t = load_data()

clf = SVC()
cs = [1000,100,10,1,0.1]
kernels = ["rbf", "sigmoid", "poly"]

param_grid = dict(	C=cs,
                    kernel=kernels)

grid = GridSearchCV(estimator=clf, param_grid=param_grid,  n_jobs=4)
grid_result = grid.fit(x_v, y_v)
write_results(grid_result.best_params_, grid_result.best_score_)