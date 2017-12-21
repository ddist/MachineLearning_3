import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

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

with open("PP Tree", "w") as file:
	print("Hyperparameter validation for MinMax scaled data\n", file=file)

def write_results(best_params, best_score):
	with open("PP Tree", "a") as file:
		for (k,v) in best_params.items():
		    print("{:<20}{}".format(k, v), file=file)
		print("{:<20}{}".format("SVM BEST SCORE ", best_score), file=file)

x_tr, x_v, x_t, y_tr, y_v , y_t = load_data()

mm_scaler = MinMaxScaler()
x_tr_mm = mm_scaler.fit_transform(x_tr)
x_v_mm = mm_scaler.transform(x_v)
x_t_mm = mm_scaler.transform(x_t)

clf = DecisionTreeClassifier()
criterions = ["gini", "entropy"]
splitters = ["best", "random"]
depths = [5,10,20,40]

param_grid = dict(	criterion=criterions,
                    splitter=splitters,
                    max_depth=depths)

grid = GridSearchCV(estimator=clf, param_grid=param_grid,  n_jobs=4)
grid_result = grid.fit(x_v_mm, y_v)
write_results(grid_result.best_params_, grid_result.best_score_)