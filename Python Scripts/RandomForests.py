from  sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from plottingFunctions import plot_decision_regions
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                 header=None)

label_enc = LabelEncoder()
X = df.iloc[:,2:4].values
y = df.iloc[:,4].values
label_enc.fit(y)
y = label_enc.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

forest = RandomForestClassifier(criterion='entropy', n_estimators=10, n_jobs=2, random_state=1)
forest.fit(X_train, y_train)

X_combined = np.vstack((X_train,X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined, y=y_combined, classifier=forest,
                      test_idx=(np.size(X_train,0)-1, np.size(X_combined,0)-1))




