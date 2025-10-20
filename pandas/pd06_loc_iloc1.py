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
# print(df)
#      종목명    시가    종가
# 031   삼성  1000  2000
# 059   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000
# 023  네이버   100  1500

# print(df[0])
# print(df['031'])
# print(df['종목명'])
# !!!판다스 열행!!!

##### 아모레 시가 출력 #####

# print(df[3, 1])             # KeyError: (3, 1)
# print(df['045', '종목명'])  # KeyError: ('045', '종목명')
# print(df['045']['종목명'])  # KeyError: '045'

# print(df['종목명']['045'])  # 아모레   # !!!! 판다스 열행 !!!!
# print(df[3][1])             # KeyError: 3

# pandas 에는 인덱스가 없는 게 절대 없음 인덱스를 떼어내도 자동 행을 넣어줌

# loc : 인덱스 기준으로 행 데이터 추출
# iloc : 행번호를 기준으로 행 데이터 추출 (자동 인덱스) -> int location 로 외워라!!!

print("====== 아모레 뽑기 ======")
print(df.iloc[3])
# 종목명     아모레
# 시가     3500
# 종가     6000
# print(df.iloc['045'])         # TypeError: Cannot index by location index with a non-integer key
# print(df.loc[3])              # KeyError: 3
print(df.loc['045'])
# ====== 아모레 뽑기 ======
# 종목명     아모레
# 시가     3500
# 종가     6000

# 판다스는 열행이고
# loc, iloc은 행열임

# 판다스는 열행이고
# loc, iloc는 행열임

print("========== 아모레 종가 뽑기 ===========")
# print(df.iloc[3][2])                # 6000
# print(df.iloc[3]['종가'])           # 6000
# print(df.iloc[3,2])                 # 6000
# print(df.iloc[3,'종가'])            #ValueError: Location based indexing can only have [integer, integer slice (START point is INCLUDED, END point is EXCLUDED), listlike of integers, boolean array] types
    
# print(df.loc['045'][2])             # 6000
# print(df.loc['045']['종가'])        # 6000
# print(df.loc['045',2])              # KeyError: 2
# print(df.loc['045', '종가'])        # 6000

# print(df.iloc[3].iloc[2])           # 6000
# print(df.iloc[3].loc['종가'])       # 6000
# print(df.loc['045'].loc['종가'])    # 6000
# print(df.loc['045'].iloc[2])        # 6000

print('========== 아모레와 네이버 시가 =======')
# print(df.iloc[3:,1])
# 045    3500
# 023     100
# print(df.iloc[3:][1])           # KeyError: 1
print()
# print(df.loc['045':]['시가'])
# 045    3500
# 023     100
# print(df.loc['045':][1])        # KeyError: 1

print(df.iloc[3:5].loc['시가'])

