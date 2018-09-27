# Importing the libraries
import numpy as np
import pandas as pd
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
df_train = pd.read_csv('dataset/train.csv')
df_test = pd.read_csv('dataset/test.csv')
df_train = df_train[sorted(df_train.columns)]
df_test = df_test[sorted(df_test.columns)]

s = df_train.isnull().sum()
t = df_test.isnull().sum()

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
df_test = df_test.drop([1116], axis = 0)
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

    





# Rows with missing data
# df_test['BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF']
# df_test['GarageQual', 'GarageFinish', 'GarageCond', 'GarageYrBlt', 'GarageArea', 'GarageCars'][1116]