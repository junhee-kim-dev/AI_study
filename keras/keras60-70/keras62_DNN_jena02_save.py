# jena 데이터 자르기

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import time

path = './_data/kaggle/jena/'
clean_jena_csv = pd.read_csv(path + 'jena_cleaned_final.csv', index_col=0)
jena_csv = pd.read_csv(path + 'jena_climate_2009_2016.csv')

submit_csv = jena_csv[['Date Time', 'wd (deg)']]
submit_csv = submit_csv[-144:]
submit_csv.to_csv(path+ 'submission_file.csv', index=False)

steps = 144
# print(jena.columns)
# Index(['Date Time', 'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)',
#        'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
#        'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
#        'wd (deg)'],
#       dtype='object')
# jena = jena.replace(-9999.000000, np.nan)
# # print(jena.describe())
# jena = jena.fillna(jena.median())

# print(jena.shape)   (420551, 14)

#### x, y######
jena_xy = clean_jena_csv[:-144]
# print(jena_xy)      #[420263 rows x 14 columns]

# jena_xy_drop = jena_xy.drop(["p (mbar)", "T (degC)", "rh (%)", "rho (g/m**3)", "wv (m/s)"], axis=1)
all_test= clean_jena_csv[-288:-144]
lets_test = all_test['wd (deg)']
test= all_test.drop(['wd (deg)'], axis=1)
# exit()
jena_xy_drop = np.array(jena_xy)


def split_1d(dataset, timesteps, stride):
    all_x, all_y = [], []
    for i in range(0,len(dataset) - timesteps -144 +1, stride) :
        x_subset = dataset[i : (i+timesteps), 0:-1]
        all_x.append(x_subset)
        y_subset = dataset[i:i+timesteps,-1]
        all_y.append(y_subset)
    # all_x_con = np.concatenate(all_x, axis=0)
    # all_y_con = np.concatenate(all_y, axis=0)
    x = np.array(all_x)
    y = np.array(all_y)
    return x, y

s_time=time.time()
print('##########xysplit 시작##########')
x, y = split_1d(jena_xy_drop, steps, 36)
# print(x)
# print(x.shape)          #(11677, 144, 8)
# print(y)
# print(y.shape)          #(11677, 144)
# print(test)    
# print(test.shape)       #(144, 8)
# exit()
e1_time=time.time()

print('##########testsplit 시작##########')
# test = 

# test, lets_test = split_1d(jena_test, steps)
# print(test)
# print(test.shape)   #(144, 14)
# # print(lets_test)
# print(lets_test.shape)  #(144,)
# exit()
e2_time=time.time()
print('##########저장 시작##########')
# x = x.astype(np.float64)
# test = test.astype(np.float64)
# print(x.dtype)
# print(y.dtype)
# print(test.dtype)
# print(lets_test.dtype)


# exit()
npy_path = './_data/kaggle/jena/dataset/'
np.save(npy_path + '2_clean_feat_(144steps_36st)x.npy', arr=x)
np.save(npy_path + '2_clean_feat_(144steps_36st)y.npy', arr=y)
np.save(npy_path + '2_clean_feat_(144steps_36st)test.npy', arr=test)
np.save(npy_path + '2_clean_feat_(144steps_36st)lets_test.npy', arr=lets_test)

e3_time=time.time()
print('##########저장 끝##########')

print('xy_split  :', np.round(e1_time- s_time, 1), 'sec')
print('test_split:', np.round(e2_time- e1_time, 1), 'sec')
print('save_time :', np.round(e3_time- e2_time, 1), 'sec')