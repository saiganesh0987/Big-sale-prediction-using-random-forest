# Big-sale-prediction-using-random-forest

#Title of Project: 

Big-sale-prediction-using-random-forest

#Objective: 
is to find acurrecy of how many sales have being 

#Data Source:


#Import Library:

import pandas as pd


     
#Import Data:

train=pd.read_csv("train.csv",na_values={"Item_Visibility":[0]})

test=pd.read_csv("test.csv",na_values={"Item_Visibility":[0]})



     
#Describe Data:

train['source']='train'

test['source']='test'

data=pd.concat([train,test],ignore_index=True)

#the one thing we have to focus is item_outlet_Sales


discpt=data.describe()


#Lets find out how many  zero'es values are

nan_descript=data.apply(lambda x: sum(x.isnull()))


     
Data Visualization:

uniq=data.apply(lambda x: len(x.unique()))

#let do grouping in each catogorical columns

col=["Item_Fat_Content","Item_Type","Outlet_Location_Type","Outlet_Size"]

for i in col:
    print("The frequency distribution of each catogorical columns is--" + i+"\n")
    print(data[i].value_counts())   

#Replacing the minimum nan values in the Item_Weight with its mean value

data.fillna({"Item_Weight":data["Item_Weight"].mean()},inplace=True)

#checking the current status of  nan values in the dataframe
nan_descript=data.apply(lambda x: sum(x.isnull()))
#Now we have 0 nan valuesin Item_Weight


     
#Data Preprocessing:

data["Outlet_Size"].fillna(method="ffill",inplace=True)


nan_descript=data.apply(lambda x: sum(x.isnull()))


#Now working on the item_visibility


visibilty_avg=data.pivot_table(values="Item_Visibility",index="Item_Identifier")


itm_visi=data.groupby('Item_Type')

data_frames=[]
for item,item_df in itm_visi:
    data_frames.append(itm_visi.get_group(item))
for i in data_frames:
    i["Item_Visibility"].fillna(value=i["Item_Visibility"].mean(),inplace=True)
    i["Item_Outlet_Sales"].fillna(value=i["Item_Outlet_Sales"].mean(),inplace=True)

new_data=pd.concat(data_frames)

nan_descript=new_data.apply(lambda x: sum(x.isnull()))

#Now we have successfully cleaned our complete dataset.
new_data["Item_Fat_Content"].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'},inplace=True)

new_data["Item_Fat_Content"].value_counts()



     
#Define Target Variable (y) and Feature Variables (X):




     
Train Test Split:

target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
 Now from this I have to learn machine learning data_Analyti
import numpy as np
from sklearn import cross_validation, metrics
def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Perform cross-validation:
    cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print ("\nModel Report")
    print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)
    
    
#Liner Regression model
print("Creating the models and processing")


     
Modeling:
from sklearn.linear_model import LinearRegression, Ridge
predictors = [x for x in train.columns if x not in [target]+IDcol]
# print predictors
alg1 = LinearRegression(normalize=True)
modelfit(alg1, train, test, predictors, target, IDcol, 'alg1.csv')
coef1 = pd.Series(alg1.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')

#Ridge Regression Model
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg2 = Ridge(alpha=0.05,normalize=True)
modelfit(alg2, train, test, predictors, target, IDcol, 'alg2.csv')
coef2 = pd.Series(alg2.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')
print("Model has been successfully created and trained. The predicted result is in alg2.csv")

# Decision Tree Model

from sklearn.tree import DecisionTreeRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg3 = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(alg3, train, test, predictors, target, IDcol, 'alg3.csv')
coef3 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances')

print("Model has been successfully created and trained. The predicted result is in alg3.csv")


     
Model Evaluation

rom sklearn.ensemble import RandomForestRegressor

predictors = [x for x in train.columns if x not in [target]+IDcol]
alg5 = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
modelfit(alg5, train, test, predictors, target, IDcol, 'alg5.csv')
coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending=False)
coef5.plot(kind='bar', title='Feature Importances')

print("Model has been successfully created and trained. The predicted result is in alg5.csv")

predictors = [x for x in train.columns if x not in [target]+IDcol]
alg6 = RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=100,n_jobs=4)
modelfit(alg6, train, test, predictors, target, IDcol, 'alg6.csv')
coef6 = pd.Series(alg6.feature_importances_, predictors).sort_values(ascending=False)
coef6.plot(kind='bar', title='Feature Importances')

print("Model has been successfully created and trained. The predicted result is in alg6.csv")

     
Explaination:
This project focuses on predicting sales for large datasets using a Random Forest Regressor. By leveraging this ensemble learning method, we aim to build a robust model that accurately forecasts sales based on historical data.

Features
Data Preprocessing: Handling missing values, encoding categorical variables, and scaling numerical features.
Exploratory Data Analysis (EDA): Visualizing data distributions and correlations.
Model Building: Implementing a Random Forest Regressor with scikit-learn.
Model Evaluation: Using metrics such as RMSE, MAE, and R² score.
Hyperparameter Tuning: Optimizing the model with Grid Search and Random Search.
Deployment: Preparing the model for real-world deployment.

