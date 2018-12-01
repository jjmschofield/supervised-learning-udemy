import os, inspect
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


def get_dataset():
    path = os.path.dirname(os.path.abspath(__file__)) # inspect.stack()[0][1]  #os.path.dirname(os.path.realpath(__file__))
    file_path = path + '/Churn_Modelling.csv'
    dataset = pd.read_csv(file_path)
    return dataset


def split_dataset(dataset):
    x = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values
    return x, y


def encode_independent_vars(x):
    geo_encoder = LabelEncoder()
    x[:, 1] = geo_encoder.fit_transform(x[:, 1])  # Encode geo
    gender_encoder = LabelEncoder()
    x[:, 2] = gender_encoder.fit_transform(x[:, 2])  # Encode gender

    # onehotencoder = OneHotEncoder(categorical_features=[1])  # TODO - Do something I don't really understand?
    onehotencoder = OneHotEncoder(categorical_features = [1])  # TODO - Do something I don't really understand?
    x = onehotencoder.fit_transform(x).toarray()
    x = x[:, 1:]  # remove one column??? Dummy variable???

    return x


def split_training_set(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=0)


def scale_features(x_train, x_test):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    return x_train, x_test, sc
