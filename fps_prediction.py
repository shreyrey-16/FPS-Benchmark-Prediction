import pandas as pd
df = pd.read_csv('fps_benchmark.csv')
df

df['CpuNumberOfTransistors'].isnull().sum()

df['CpuName'] = df['CpuName'].astype(str).str.strip("b'")
df['GpuName'] = df['GpuName'].astype(str).str.strip("b'")
df['GpuArchitecture'] = df['GpuArchitecture'].astype(str).str.strip("b'")
df['GpuBus.interface'] = df['GpuBus.interface'].astype(str).str.strip("b'")
df['GpuDirectX'] = df['GpuDirectX'].astype(str).str.strip("b'")
df['GpuMemorySize'] = df['GpuMemorySize'].astype(str).str.strip("b'")
df['GpuMemoryType'] = df['GpuMemoryType'].astype(str).str.strip("b'")
df['GpuOpenCL'] = df['GpuOpenCL'].astype(str).str.strip("b'")
df['GpuOpenGL'] = df['GpuOpenGL'].astype(str).str.strip("b'")
df['GpuShaderModel'] = df['GpuShaderModel'].astype(str).str.strip("b'")
df['GpuVulkan'] = df['GpuVulkan'].astype(str).str.strip("b'")
df['GameName'] = df['GameName'].astype(str).str.strip("b'")
df['GameSetting'] = df['GameSetting'].astype(str).str.strip("b'")
df['CpuMultiplier'] = df['CpuMultiplier'].astype(str).str.strip("b'")
df['CpuMultiplierUnlocked'] = df['CpuMultiplierUnlocked'].astype(str).str.strip("b'")

df['CpuBrand'] = df['CpuName'].str.extract(r'(Intel|AMD)')
df['GpuBrand'] = df['GpuName'].str.extract(r'(NVIDIA|AMD)')

df.isnull().sum()

df = df.drop(['CpuDieSize', 'CpuNumberOfTransistors', 'GpuNumberOfComputeUnits', 'GpuNumberOfExecutionUnits','CpuName','GpuName'], axis = 1)

df.head()

for i in df:
    print(f"{i} : {len(df[i].unique())}, {df[i].dtype}")
    
df['CpuTurboClock'].unique()

cols_to_onehotencode = ['CpuBrand','GpuBrand','GameName','GameSetting','GpuVulkan','GpuShaderModel','GpuBus.interface','GpuArchitecture','GpuOpenGL','GpuOpenCL','GpuMemoryType','GpuDirectX','CpuProcessSize','CpuMultiplierUnlocked','CpuBaseClock','GpuMemorySize','CpuMultiplier']

df_encoded = pd.get_dummies(df[cols_to_onehotencode])

df_encoded.head()

df = df.drop(columns = cols_to_onehotencode)
df.head()

df_new = pd.concat([df, df_encoded], axis = 1)

X = df_new.drop('FPS', axis = 1)
y = df_new['FPS']

y = (y - y.mean())/(y.std())

l = []
d = dict()
for i in X:
    corr = df_new[i].corr(df_new['FPS'])
    d[i] = corr
    l.append(corr)
    
d

a = range(0,80)

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
sns.kdeplot(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

params_xgb = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

params_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

grid_xgb = GridSearchCV(xgb.XGBRegressor(), params_xgb, cv = 5, scoring = 'neg_mean_squared_error')
grid_rf = GridSearchCV(RandomForestRegressor(), params_rf, cv = 5, scoring = 'neg_mean_squared_error')

grid_xgb.fit(X_train, y_train)
grid_rf.fit(X_train, y_train)

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

xgb = grid_xgb.predict(X_test)
rf = grid_rf.predict(X_test)
lr = model_lr.predict(X_test)

from sklearn.metrics import r2_score
print(f"XGBoost Regressor (R2 score) : {r2_score(y_test, xgb)*100:.2f}")
print(f"Random Forest Regressor (R2 score) : {r2_score(y_test, rf)*100:.2f}")
print(f"Linear Regression (R2 score) : {r2_score(y_test, lr)*100:.2f}")

print(grid_rf.best_params_)
print(grid_xgb.best_params_)