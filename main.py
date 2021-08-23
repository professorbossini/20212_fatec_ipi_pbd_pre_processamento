import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv(r'04_dados_aula.csv')

features = dataset.iloc[ : , :-1].values
# print (features)
classe = dataset.iloc[ : , -1 ].values
# print (classe)

imputer = SimpleImputer (missing_values=np.nan, strategy="mean")

imputer.fit(features[ : , 1:3])

features[:, 1:3] = imputer.transform(features[:, 1:3])

columnTransformer = ColumnTransformer (
    transformers=[('encoder', OneHotEncoder(), [0])],
    remainder='passthrough'
)
features = np.array(columnTransformer.fit_transform(features))
# print (features)

labelEncoder = LabelEncoder()
classe = labelEncoder.fit_transform(classe)
print (classe)


