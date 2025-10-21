import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, OneHotEncoder, PolynomialFeatures, FunctionTransformer, QuantileTransformer
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import r2_score, root_mean_squared_error
import time
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
import pickle

def get_data(file="data/Test.csv", fraction=1, drop=[], seed=42):
    df = pd.read_csv(file, parse_dates=["Date"],date_format="%Y-%m-%d" )
    df["Date"] = df["Date"].values.astype(np.int64) // 10**9
    df = df.sample(frac=fraction, random_state=seed)
    df = df.drop(drop, axis=1)
    return df

def column_threshold(dataframe,threshold,drop, excludes):
   
    columns = {"column":[], "not_null": [], "exclude":[]}
    column_list = dataframe.columns.to_list()
    for column in column_list:
        excl=False
        for exclude in excludes:
            if exclude in column:
                excl = True                
        
        columns["column"].append(column)
        columns["not_null"].append(dataframe[column].count()/dataframe['target'].count()*100)
        columns["exclude"].append(excl)

    cols=pd.DataFrame(columns)
    features_to_use=list(set(cols[(cols["not_null"]>threshold)&(cols["exclude"]==False)]["column"])-set(["target"])-set(drop))
    return features_to_use

model_name="Elastic_Net"
loaded_model = pickle.load(open(f"models/{model_name}.sav", 'rb'))

df = get_data()
target="target"
df[target]=np.log10(df[target])

prediction = loaded_model.predict()

encode_location = Pipeline([ 
        ("encode", OneHotEncoder(drop="first"))]) 
    
scale_impute = Pipeline([   
        ("imputer_null",KNNImputer(n_neighbors=10, weights="distance")), # deals with null  
        ("imputer_0",KNNImputer(n_neighbors=10, weights="distance", missing_values=0)), # deals with 0  
        ("scaler", scaler )
        ]) 
preprocess = ColumnTransformer([    
        ("scale_impute", scale_impute, features_numeric),
        ("encode_location", encode_location, features_categorical)],remainder="drop")
