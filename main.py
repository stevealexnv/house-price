# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

# Importing the dataset
orig_train = pd.read_csv('dataset/train.csv')
orig_test = pd.read_csv('dataset/test.csv')
df_train = orig_train.copy()
df_test = orig_test.copy()
#df_train = orig_train[sorted(orig_train.columns)]
#df_test = orig_test[sorted(orig_test.columns)]

#s = df_train.isnull().sum()
#t = df_test.isnull().sum()

# Filling missing data
df_train['BsmtExposure'][948] = 'No'
df_train['BsmtFinType2'][332] = 'Rec'
df_test['MasVnrType'][1150] = 'BrkFace'
df_test['BsmtCond'][1064] = 'TA'
df_test['BsmtCond'][725] = 'TA'
df_test['BsmtCond'][580] = 'TA'
df_test['BsmtExposure'][27] = 'No'
df_test['BsmtExposure'][888] = 'No'
df_test['BsmtQual'][757] = 'TA'
df_test['BsmtQual'][758] = 'TA'
df_test['GarageCond'][666] = 'TA'
df_test['GarageQual'][666] = 'TA'
df_test['GarageFinish'][666] = 'Unf'
df_test['GarageYrBlt'][666] = df_test['GarageYrBlt'].median()
df_test['GarageType'][1116] = 'No Garage'
df_data = [df_train, df_test]
for dataset in df_data:
    dataset['LotFrontage'] = dataset.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    dataset['Electrical'] = dataset['Electrical'].fillna(dataset['Electrical'].mode()[0])
    dataset['SaleType'] = dataset['SaleType'].fillna(dataset['SaleType'].mode()[0])
    dataset['Utilities'] = dataset['Utilities'].fillna(dataset['Utilities'].mode()[0])
    dataset['MSZoning'] = dataset['MSZoning'].fillna(dataset['MSZoning'].mode()[0])
    dataset['Exterior1st'] = dataset['Exterior1st'].fillna(dataset['Exterior1st'].mode()[0])
    dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna(dataset['Exterior2nd'].mode()[0])
    dataset['MasVnrType'] = dataset['MasVnrType'].fillna('None')
    dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(0)
    dataset['Alley'] = dataset['Alley'].fillna('No Access')
    dataset['Fence'] = dataset['Fence'].fillna('No Fence')
    dataset['FireplaceQu'] = dataset['FireplaceQu'].fillna('No Fireplace')
    dataset['MiscFeature'] = dataset['MiscFeature'].fillna('None')
    dataset['PoolQC'] = dataset['PoolQC'].fillna('No Pool')
    dataset['Functional'] = dataset['Functional'].fillna('Typ')
    dataset['KitchenQual'] = dataset['KitchenQual'].fillna('TA')
    dataset['GarageType'] = dataset['GarageType'].fillna('No Garage')
    dataset['GarageYrBlt'] = dataset['GarageYrBlt'].fillna(0)
    dataset['GarageFinish'] = dataset['GarageFinish'].fillna('No Garage')
    dataset['GarageQual'] = dataset['GarageQual'].fillna('No Garage')
    dataset['GarageCond'] = dataset['GarageCond'].fillna('No Garage')
    dataset['GarageCars'] = dataset['GarageCars'].fillna(0)
    dataset['GarageArea'] = dataset['GarageArea'].fillna(0)
    dataset['BsmtQual'] = dataset['BsmtQual'].fillna('No Basement')
    dataset['BsmtCond'] = dataset['BsmtCond'].fillna('No Basement')
    dataset['BsmtExposure'] = dataset['BsmtExposure'].fillna('No Basement')
    dataset['BsmtFinType1'] = dataset['BsmtFinType1'].fillna('No Basement')
    dataset['BsmtFinSF1'] = dataset['BsmtFinSF1'].fillna(0)
    dataset['BsmtFinType2'] = dataset['BsmtFinType2'].fillna('No Basement')
    dataset['BsmtFinSF2'] = dataset['BsmtFinSF2'].fillna(0)
    dataset['BsmtUnfSF'] = dataset['BsmtUnfSF'].fillna(0)
    dataset['BsmtFullBath'] = dataset['BsmtFullBath'].fillna(0)
    dataset['BsmtHalfBath'] = dataset['BsmtHalfBath'].fillna(0)
    dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].fillna(dataset['BsmtFinSF1'] + dataset['BsmtFinSF2'] + dataset['BsmtUnfSF'])

# Adding new feature TotalSF
for dataset in df_data:
    dataset['TotalSF'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']

# Encoding categorical data
cat_col = ['MSSubClass',
       'MSZoning',
       'Street',
       'Alley',
       'LotShape',
       'LandContour',
       'Utilities',
       'LotConfig',
       'LandSlope',
       'Neighborhood',
       'Condition1',
       'Condition2',
       'BldgType',
       'HouseStyle',
       'RoofStyle',
       'RoofMatl',
       'Exterior1st',
       'Exterior2nd',
       'MasVnrType',
       'ExterQual',
       'ExterCond',
       'Foundation',
       'BsmtQual',
       'BsmtCond',
       'BsmtExposure',
       'BsmtFinType1',
       'BsmtFinType2',
       'Heating',
       'HeatingQC',
       'CentralAir',
       'Electrical',
       'KitchenQual',
       'Functional',
       'FireplaceQu',
       'GarageType',
       'GarageFinish',
       'GarageQual',
       'GarageCond',
       'PavedDrive',
       'PoolQC',
       'Fence',
       'MiscFeature',
       'SaleType',
       'SaleCondition']
df = df_train.append(df_test, sort = False)
for c in cat_col:
    le = LabelEncoder()
    le.fit(df[c].values)
    for dataset in df_data:
        dataset[c] = le.transform(dataset[c].values)

# Feature Selection
ID = df_test['Id']
drop_features = ['Utilities', 'Id']
df_train = df_train.drop(drop_features, axis = 1)
df_test = df_test.drop(drop_features, axis = 1)

# Creating Training and Test set arrays
X_test = df_test.values
y_train = df_train['SalePrice'].values
df_train = df_train.drop(['SalePrice'], axis = 1)
X_train = df_train.values

# Using One Hot Encoder to remove ordering of classes in


# Avoiding dummy variable trap


#  Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1, 1))
y_train = y_train.ravel()

# Fitting different regressors
svm = SVR()
svm.fit(X_train, y_train)
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
xgb = XGBRegressor()
xgb.fit(X_train, y_train)

# Applying k-Fold Cross Validation to all regressors
col = ['Regressor', 'Mean', 'Standard Deviation']
accuracy = pd.DataFrame(index = range(0,4), columns = col)
accuracy['Regressor'] = ['SVM',
                        'Decision Tree',
                        'Random Forest',
                        'XGBoost']
acc_mean = []
acc_std = []
score_svm = cross_val_score(estimator = svm, X = X_train, y = y_train, cv = 10)
acc_mean.append(score_svm.mean())
acc_std.append(score_svm.std())
score_dt = cross_val_score(estimator = dt, X = X_train, y = y_train, cv = 10)
acc_mean.append(score_dt.mean())
acc_std.append(score_dt.std())
score_rf = cross_val_score(estimator = rf, X = X_train, y = y_train, cv = 10)
acc_mean.append(score_rf.mean())
acc_std.append(score_rf.std())
score_xgb = cross_val_score(estimator = xgb, X = X_train, y = y_train, cv = 10)
acc_mean.append(score_xgb.mean())
acc_std.append(score_xgb.std())
accuracy['Mean'] = acc_mean
accuracy['Standard Deviation'] = acc_std
print(accuracy)

# Predicting the Test set results
y_pred = sc_y.inverse_transform(xgb.predict(X_test))

# Generating submission file
submission = pd.DataFrame({'Id': ID, 'SalePrice': y_pred})
submission.to_csv('result.csv', index=False)