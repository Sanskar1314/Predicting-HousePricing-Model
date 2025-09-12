import pandas as pd
import numpy as np
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

#1.  load the data

housing = pd.read_excel("housing.xlsx") 

#2. Create a stratified test set based on income category

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0,1.5,3.0,4.5,6.0,np.inf],
                               labels=[1,2,3,4,5])
        
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat",axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat",axis=1)


#3. we will work on the copy of training set

housing = strat_train_set.copy()

#4. separate features and labels

housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value",axis=1)

#5. separating numerical and categorical columns

num_attributes = housing.drop("ocean_proximity",axis=1).columns.tolist()
cat_attributes = ["ocean_proximity"]

#6. constructing pipelines for numerical and categorical attributes

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
    ("num",num_pipeline,num_attributes),
    ("cat",cat_pipeline,cat_attributes)
])

#8. transforming the data

housing_prepared = full_pipeline.fit_transform(housing)

# housing_prepared is now in a numppy array format and is ready for training
print(housing_prepared.shape)

#9. Training the model

# linear regression model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
lin_rmse = root_mean_squared_error(housing_labels,lin_preds)
#print(f"Rmse for linear regression is {lin_rmse}")
# evaluating linear regression model using cross validation
lin_rmses = -cross_val_score(lin_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)   
print(pd.Series(lin_rmses).describe())
# DecisionTree regression model
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared,housing_labels)
dec_preds = dec_reg.predict(housing_prepared)
# dec_rmse = root_mean_squared_error(housing_labels,dec_preds)
#print(f"Rmse for DecisionTree regression is {dec_rmse}")
# evaluating DecisionTree regression model using cross validation
dec_rmses = -cross_val_score(dec_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)   
print(pd.Series(dec_rmses).describe())
# RandomForest regression model
ran_reg = RandomForestRegressor()
ran_reg.fit(housing_prepared,housing_labels)
ran_preds = ran_reg.predict(housing_prepared)
# ran_rmse = root_mean_squared_error(housing_labels,ran_preds)
# print(f"Rmse for RandomForest  regression is {ran_rmse}")
# evaluating RandomForest regression model using cross validation
ran_rmses = -cross_val_score(ran_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)   
print(pd.Series(ran_rmses).describe())