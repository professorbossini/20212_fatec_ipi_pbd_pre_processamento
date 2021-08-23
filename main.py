import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


features_treinamento, features_teste, classe_treinamento, classe_teste = train_test_split(
    features, classe, test_size=0.2, random_state=1
)

# print (features_treinamento)
# print ('----------------')
# print (features_teste)
# print ('----------------')
# print (classe_treinamento)
# print ('----------------')
# print (classe_teste)
print (features_treinamento)
standardScaler = StandardScaler()
features_treinamento[:, 3: ] = standardScaler.fit_transform(features_treinamento[ : , 3: ])
print (features_treinamento)

features_teste[:, 3: ] = standardScaler.transform(features_teste[:, 3:])
print (features_teste)