
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import gc

from fbprophet import Prophet
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

sns.set_style("dark")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


train_df = pd.read_csv("train.csv",parse_dates=['date'])
test_df = pd.read_csv("test.csv",parse_dates=['date'])

train_df['train_or_test'] = 'train'
test_df['train_or_test'] = 'test'
test_df['sales'] = np.nan
df = pd.concat([train_df, test_df.loc[:, ['train_or_test','store', 'item', 'sales','date']]], sort=False, ignore_index=True)
del train_df
del test_df
gc.collect()

df['year'] = df.date.dt.year
df['month'] = df.date.dt.month
df['dayofmonth'] = df.date.dt.day
df['dayofweek'] = df.date.dt.dayofweek
df['dayofyear'] = df.date.dt.dayofyear
df['weekofyear'] = df.date.dt.weekofyear
df['is_month_start'] = (df.date.dt.is_month_start).astype(int)
df['is_month_end'] = (df.date.dt.is_month_end).astype(int)
df['quarter'] = df.date.dt.quarter
df['week_block_num'] = [int(x) for x in np.floor((df.date - pd.to_datetime('2012-12-31')).dt.days/7) + 1]
df['quarter_block_num'] = (df['year'] - 2013) * 4 + df['quarter']


df.sort_values(by=['store','item','date'], axis=0, inplace=True)

df.month.isin([1,2,3])















train_df = df[df.train_or_test=='train']
test_df = df[df.train_or_test=='test']
train_x = train_df.loc[:,[col for col in train_df.columns if col not in ['sales','train_or_test','date']]]
train_y = train_df['sales']
test_x = test_df.loc[:,[col for col in test_df.columns if col not in ['sales','train_or_test','date']]]

xgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'reg:linear',
    'eval_metric': '',
    'learning_rate': 0.02,
    'verbose': 0,
    'num_leaves': 1024,
    'max_depth' : 16,
    'max_bin': 255,
    }

x_train, x_valid, y_train, y_valid = train_test_split(train_x,train_y,test_size=0.25,random_state=42)

xgb_dtrain = xgb.DMatrix(x_train, label=y_train)
xgb_dvalid = xgb.DMatrix(x_valid, label=y_valid)


evals_results = {}
print("Training the xgb model...")
watchlist = [(xgb_dtrain, 'train'),(xgb_dvalid, 'valid')]

start = datetime.now()
xgb_model = xgb.train(xgb_params, 
                 xgb_dtrain,
                 2000,
                 watchlist,
                 evals_result=evals_results, 
                 early_stopping_rounds=35,
                 verbose_eval=True)
print("Total time taken : ", datetime.now()-start)


xgb_x = xgb_model.predict(xgb.DMatrix(test_x),ntree_limit=xgb_model.best_iteration)

xgb_x = pd.DataFrame(xgb_x)
answer = xgb_x.reset_index()
answer.rename(columns={'index' : 'id',0 : 'sales'},inplace=True)


prophet_train_df = pd.DataFrame()
prophet_test_df = pd.DataFrame()
prophet_train_df['ds'] = train_df['date']
prophet_test_df['ds'] = test_df['date']
prophet_train_df['y'] = train_y


xxxx = prophet_train_df['ds']

m = Prophet()
start = datetime.now()
m.fit(prophet_train_df)
future = m.make_future_dataframe(periods=90).tail(90)
forecast = m.predict(future)
print("Total time taken : ", datetime.now()-start)

answer3 = pd.DataFrame()
for i in range(0,forecast.shape[0]):
    for j in range(0,answer.shape[0]):
        if(forecast.at[i,'ds'] == test_df.at[j,'date']):
            answer3.at[j,'sales'] = 0.7*answer.at[j,'sales'] + 0.3*forecast.at[i,'yhat']

answer['sales'] = answer3['sales']

m.plot(forecast)
answer.to_csv('submission_xgb_basic.csv',index=False)