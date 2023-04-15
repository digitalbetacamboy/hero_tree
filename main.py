# Библиотеки
import warnings
import re
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.model_selection \
import cross_val_score
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

medbase = pd.read_excel('medbase.xlsx')
rez = pd.read_excel('rez.xlsx')

medbase.drop(labels=None, axis=1, columns=['id','user','v01','v02','v03','v04','dr','v05','page','poputka'], inplace=True)
rez.drop(labels=None, axis=1, columns=['id','user','id_pr'], inplace=True)

X = medbase.copy()
Y = rez.copy()

# 5) разделение набора на тренировочный и тестовый
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# 6) создание и тренировка модели (1000 деревьев)
model = RandomForestClassifier(n_estimators=2000, n_jobs=4, criterion='entropy')
model.fit(X_train, y_train)

from sklearn.metrics import f1_score
print(f1_score(y_test.to_numpy().ravel(), model.predict(X_test).ravel(), average='weighted'))

from sklearn.metrics import precision_score
print(precision_score(y_test.to_numpy().ravel(), model.predict(X_test).ravel(), average='weighted'))