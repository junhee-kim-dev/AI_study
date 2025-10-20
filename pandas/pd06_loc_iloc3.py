import numpy as np
import pandas as pd

print(pd.__version__)

data = [
    ['삼성', '1000', '2000'],
    ['현대', '1100', '3000'],
    ['LG', '2000', '500'],
    ['아모레', '3500', '6000'],
    ['네이버', '100', '1500'],
]

index = ['031', '059', '033', '045', '023']
columns = ['종목명', '시가', '종가']

df = pd.DataFrame(data=data, index=index, columns=columns)

print("=====================")
print("시가가 1000원 이상인 행을 모두 출력!!")

# print(df.iloc[:, '시가'] >= 1000)

cond1 = df['시가'] >='1100'
print(df[cond1])

df3 = df[df['시가']>='1100']['종가']
print(df3)





