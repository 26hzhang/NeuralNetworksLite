#!/usr/local/bin/python3
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import make_scorer
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import tflearn
import tensorflow as tf
import seaborn
import warnings
warnings.filterwarnings('ignore')

'''
Pre-process is referred from: https://www.kaggle.com/miguelangelnieto/pca-and-regression/notebook/notebook
'''

train = pd.read_csv('train.csv')
labels = train["SalePrice"]
test = pd.read_csv('test.csv')
data = pd.concat([train, test], ignore_index=True)
data = data.drop("SalePrice", 1)
ids = test['Id']

# Remove id and columns with more than a thousand missing values
data=data.drop("Id", 1)
data=data.drop("Alley", 1)
data=data.drop("Fence", 1)
data=data.drop("MiscFeature", 1)
data=data.drop("PoolQC", 1)
data=data.drop("FireplaceQu", 1)

# Count the column types
all_columns = data.columns.values
non_categorical = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch","PoolArea", "MiscVal"]
categorical = [value for value in all_columns if value not in non_categorical]

# One Hot Encoding and nan transformation
data = pd.get_dummies(data)
imp = Imputer(missing_values="NaN", strategy="most_frequent", axis = 0)
data = imp.fit_transform(data)

data = np.log(data)
labels = np.log(labels)
data[data==-np.inf]=0

# save processed data
#np.savetxt("train.txt", train, delimiter=",")
#np.savetxt("test.txt", test, delimiter=",")
#np.savetxt("label.txt", labels, delimiter=",")

#pca = PCA(whiten=True)
#pca.fit(data)
#variance = pd.DataFrame(pca.explained_variance_ratio_)
#var = np.cumsum(pca.explained_variance_ratio_)

# PCA
pca = PCA(n_components=36,whiten=True)
pca = pca.fit(data)
dataPCA = pca.transform(data)

train = dataPCA[:1460]
test = dataPCA[1460:]

cv = KFold(n_splits=5,shuffle=True,random_state=45)

parameters = {'alpha': [1000,100,10],
              'epsilon' : [1.2,1.25,1.50],
              'tol' : [1e-10]}

clf = linear_model.HuberRegressor()
r2 = make_scorer(r2_score)
grid_obj = GridSearchCV(clf, parameters, cv=cv,scoring=r2)
grid_fit = grid_obj.fit(train, labels)
best_clf = grid_fit.best_estimator_ 

best_clf.fit(train,labels)

labels_nl = labels
labels_nl = labels_nl.reshape(-1,1)

tf.reset_default_graph()
r2 = tflearn.R2()
net = tflearn.input_data(shape=[None, train.shape[1]])
net = tflearn.fully_connected(net, 30, activation='linear')
net = tflearn.fully_connected(net, 10, activation='linear')
net = tflearn.fully_connected(net, 1, activation='linear')
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.01, decay_step=100)
net = tflearn.regression(net, optimizer=sgd,loss='mean_square',metric=r2)
model = tflearn.DNN(net)

model.fit(train, labels_nl,show_metric=True,validation_set=0.2,shuffle=True,n_epoch=50)

predictions_huber = best_clf.predict(test)
predictions_DNN = model.predict(test)
predictions_huber = np.exp(predictions_huber)
predictions_DNN = np.exp(predictions_DNN)
predictions_DNN = predictions_DNN.reshape(-1,)
	
sub = pd.DataFrame({
        "Id": ids,
        "SalePrice": predictions_DNN
    })

sub.to_csv("prices_submission.csv", index=False)
print(sub)