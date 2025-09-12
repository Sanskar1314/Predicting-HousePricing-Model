import os
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

MODEL_FILE = "model.pkl"
PIPELIINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs,cat_attribs):
    # for numerical pipeline
    num_pipeline = Pipeline([
        ("imputer",SimpleImputer(strategy="mean")),
        ("scaler",StandardScaler()),
    ])

    # for categorical pipeline
    cat_pipeline = Pipeline([
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])

    #7. creating full pipeline

    full_pipeline = ColumnTransformer([
        ("num",num_pipeline,num_attribs),
        ("cat",cat_pipeline,cat_attribs)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    #train the data
    housing = pd.read_excel("housing.xlsx")
    # stratified split 
    housing['income_cat'] = pd.cut(housing["median_income"],bins=[0.0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index,test_index in split.split(housing,housing["income_cat"]):
        housing.loc[test_index].drop("income_cat",axis=1).to_excel("input.xlsx",index=False)
        housing=housing.loc[train_index].drop("income_cat",axis=1)

    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value",axis=1)

    num_attribs = housing_features.drop("ocean_proximity",axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    pipeline = build_pipeline(num_attribs,cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor()
    model.fit(housing_prepared,housing_labels)

    # save the model an pipeline
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELIINE_FILE)

    print("Model is trained and saved")    

else:
    #inference phase
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELIINE_FILE)
    
    input_data = pd.read_excel("input.xlsx")
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data['media_house_value_pred'] = predictions
    input_data.to_excel("output.xlsx",index=False)
    print("Predictions are saved to output.xlsx")   

