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
import statsmodels.formula.api as sm
from sklearn.model_selection import cross_validate, cross_val_score

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

'''# Encoding categorical data
cat_cols = ['MSSubClass',
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
for c in cat_cols:
    le = LabelEncoder()
    le.fit(df[c].values)
    for dataset in df_data:
        dataset[c] = le.transform(dataset[c].values)'''

# Feature Selection
ID = df_test['Id']
drop_features = ['Utilities', 'Id']
df_train = df_train.drop(drop_features, axis = 1)
df_test = df_test.drop(drop_features, axis = 1)
dum_cols = ['MSSubClass',
       'MSZoning',
       'Street',
       'Alley',
       'LotShape',
       'LandContour',
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
for dum_col in dum_cols:
    df = pd.concat([df, pd.get_dummies(df[dum_col], prefix = dum_col, drop_first = True)], axis = 1).drop([dum_col], axis=1)

# Creating Training and Test set arrays
df_train = df.iloc[:1460]
df_test = df.iloc[1460:]
y_train = df_train['SalePrice'].values
df_train = df_train.drop('SalePrice', axis = 1)
df_test = df_test.drop('SalePrice', axis = 1)
X_test = df_test.values
X_train = df_train.values

'''# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1, 1))
y_train = y_train.ravel()'''

# Fitting different regressors
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
xgb = XGBRegressor()
xgb.fit(X_train, y_train)

# Applying k-Fold Cross Validation to all regressors
col = ['Regressor', 'Mean', 'Standard Deviation']
rmsle = pd.DataFrame(index = range(0,3), columns = col)
rmsle['Regressor'] = ['Decision Tree',
                     'Random Forest',
                     'XGBoost']
rmsle_mean = []
rmsle_std = []
score_dt = cross_val_score(estimator = dt, X = X_train, y = y_train, scoring = 'neg_mean_squared_log_error', cv = 10)
rmsle_mean.append(score_dt.mean())
rmsle_std.append(score_dt.std())
score_rf = cross_val_score(estimator = rf, X = X_train, y = y_train, scoring = 'neg_mean_squared_log_error', cv = 10)
rmsle_mean.append(score_rf.mean())
rmsle_std.append(score_rf.std())
score_xgb = cross_val_score(estimator = xgb, X = X_train, y = y_train, scoring = 'neg_mean_squared_log_error', cv = 10)
rmsle_mean.append(score_xgb.mean())
rmsle_std.append(score_xgb.std())
rmsle['Mean'] = rmsle_mean
rmsle['Standard Deviation'] = rmsle_std
print(rmsle)

#Building the optimal model using Backward Elimination
def BackwardElimination(x, sl):
    num_vars = len(x[0])
    temp = np.zeros((1460, 273)).astype(int)
    for i in range(0, num_vars):
        regressor_OLS = sm.OLS(y_train, x).fit()
        max_pval = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if max_pval > sl:
            for j in range(0, num_vars - i):
                if(regressor_OLS.pvalues[j].astype(float) == max_pval):
                    temp[:, j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y_train, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if(adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:, [0, j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print(regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    print(regressor_OLS.summary())
    return x

SL = 0.05
X_train = np.append(arr = np.ones((1460, 1)).astype(int), values = X_train, axis = 1)
X_opt = X_train[:, :273]
X_modelled = BackwardElimination(X_opt, SL)

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y_train, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print(regressor_OLS.summary())
    return x
 
SL = 0.05
X_train = np.append(arr = np.ones((1460, 1)).astype(int), values = X_train, axis = 1)
X_opt = X_train[:, :273]
X_modelled = backwardElimination(X_opt, SL)

# Predicting the Test set results
y_pred = xgb.predict(X_test)

# Generating submission file
submission = pd.DataFrame({'Id': ID, 'SalePrice': y_pred})
submission.to_csv('result.csv', index=False)