# 열 추가하기
# from numpy import subtract
import pandas as pd
import seaborn as sns

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
# exam_data = {'이름' : [ '서준', '우현', '인아'],
#              '수학' : [ 90, 80, 70],
#              '영어' : [ 98, 89, 95],
#              '음악' : [ 85, 95, 100],
#              '체육' : [ 100, 90, 90]}

# df = pd.DataFrame(exam_data)
# print(df)
# print('\n')

# df['국어'] = 50     # 열을 추가하게 된다면 값은 모두 같게 된다.
# print(df) 

# 행 추가하기
# 하나의 데이터 입력, 열에 맞추어 여러개의 값을 입력

# df.loc[6] = 0           # loc으로 3이 아닌 그냥 값을 넣어도 된다. 그냥 인덱스만 바뀌는 것 뿐이다.
# print(df)
# print('\n')

# df.loc[4] = ['준하', 90, 80, 70, 60]
# print(df)
# print('\n')

# df.loc['행5'] = df.loc[6]       # 열의 값이 갖게 된다.
# print(df)

# 원소 값 변경하기
# 여러개의 원소를 한꺼번에 혹은 하나만 변경 가능하다.

# df.set_index('이름', inplace=True)
# print(df)
# print('\n')

# df.iloc[0][3] = 80  # 인덱스 부분의 0부터 시작된다.
# print(df)
# print('\n')

# df.loc['서준']['체육'] = 20
# print(df)
# print('\n')

# df.loc['서준','체육'] = 30
# print(df)

# df.loc['서준']['음악','체육'] = 50      # 서준의 음악, 체육의 점수가 50으로 변경된다.       , 슛저거 어나므로 loc 를 사용한다.
# print(df)
# print('\n')

# df.loc['서준', ['음악', '체육']] = 100, 50
# print(df)


# 행, 열의 위치 바꾸기(Transpose of A Matrix)       참고로 axis 는  행과 열 삭제할 때 사용한다.


# df = df.transpose() # 변환
# df = df.T       # T는 괄호를 붙이면 안된다.
# print(df)


# 인덱스 활용 

# df = df.set_index(['이름'])    # 특정 열을 행 인덱스 로 활용하기 이때, 소괄호 부분에 대괄호가 없어도 되고, 있어도 된다.
# print(df)
# print('\n')
# df2 = df.set_index(['음악'])    # 이때, 이름의 인덱스는 행에서 사라진다.
# print(df2)
# print('\n')
# df3 = df.set_index(['수학', '음악'])    # 두개 이상을 행으로 지정할 경우에는 중괄호 표기 해줘야한다.
# print(df3)
# print('\n')

# 행 인덱스 재 배열하기

# dict_data = {'c0':[1,2,3], 'c1':[4,5,6], 'c2':[7,8,9], 'c3':[10,11,12], 'c4':[13,14,15]}

# df = pd.DataFrame(dict_data , index=['r0', 'r1', 'r2'])
# print(df)
# # print(df)
# print('\n')

# df1 = pd.DataFrame(dict_data, index=['r0', 'r1', 'r2'])  # 행 인덱스를 새롭게 재배열, set_index는 열의 인덱스를 행 인덱스로 가져오는 것 set_index 의 배열은 함수이지만, index 는 표기이다.
# print(df1)
# print('\n')

# new_index = ['a0', 'a1', 'a2']
# df2 = df.reindex(index = new_index, fill_value = 0)     # 만약에 행을 함수에 지정되어 있는 개수보다 더 많은 개수가 지정되어 있다면, 열에는 NaN 값이 들어간다
# print(df2)
# print('\n')

# 행 인덱스 초기화하기

# df = df.reset_index()           # 숫자 인덱스로 다시 초기화 시켜준다
# print(df)

# 행 인덱스를 기준으로 데이터 프레임 정렬

# ndf = df.sort_index(ascending=False)    # 행 인덱스 뿐만 아니라 열 인덱스도 내림차순으로 변경된다.
# print(ndf)
# print('\n')
# # 특정열의 데이터 값을 기준으로 데이터프레임 정렬하기

# ndf2 = df.sort_values(by='c1', ascending=False)
# print(ndf2)


# 산술 연산

# 행 열 인덱스를 기준으로 모든 원소를 정렬

# 시리즈에 숫자 연산하기

# df = pd.Series({'국어' : 100 , '영어' : 80, '수학': 90})
# print(df)
# print('\n')

# percentage = df/200     # 모든 인덱스의 값에 접근하여 값을 나누어 준다.

# print(percentage)   

# 시리즈에 시리즈 연산

# student1 = pd.Series({'국어' : 100 , '영어' : 80, '수학': 90})
# print(student1)
# print('\n')
# student2 = pd.Series({'수학' : 80 , '국어' : 90, '영어': 80})
# print(student2)
# print('\n')

# addition = student1 + student2
# subtraction = student1 - student2
# multipled = student1 / student2
# multipledivison = student1 * student2
# print(type(multipledivison))
# print('\n')

# result = pd.DataFrame([addition , subtraction, multipled, multipledivison], index=(['덧셈', '뺄셈', '곱셈', '나눗셈']))
# 계산하는 값이 열로 들어가고 inedx로 행을 처리할 수 있다.
# print(result)

# ! 연산을 할 경우, 두 시리즈의 원소의 개수가 다르거나, 시리즈의 크기가 같더라도, 인덱스 값이 다를 수 있다.   - 정상적인 연산처리를 못한다.
# 혹은 둘다 동일한 인덱싱을 가지고 있지만, NaN값 을 가지고 있는 경우가 있다.                                                        이러한 경우 모든 값은 NaN 로 나타난다. 

# ex>
# import numpy as np

# student1 = pd.Series({'국어':np.nan, '영어':80, '수학':90})         # NaN 값을 만들기 위해서는 np.nan 을 이용해야 한다
# student2 = pd.Series({'수학':80, '국어':90})

# addition = student1 + student2
# subtraction = student1 - student2
# multipled = student1 / student2
# multipledivison = student1 * student2

# result = pd.DataFrame([addition , subtraction, multipled, multipledivison], index=(['덧셈', '뺄셈', '곱셈', '나눗셈']))

# 이러한 경우 원소가 모두 NaN 값으로 나타난다

# 위에 경우를 피하기 위해서는 연산 메소드에 fill_value = 0 을 사용해주어야 한다.

# sr_add = student1.add(student2, fill_value=0)   # 덧셈
# sr_sub = student1.sub(student2, fill_value=0)   # 뺄셈
# sr_div = student1.div(student2, fill_value=0)   # 나눗셈
# sr_mul = student1.mul(student2, fill_value=0)   # 곱셈

# result = pd.DataFrame([sr_add, sr_sub, sr_mul, sr_div])
# print(result)               # 0의 나눗셈은 infinite 로 나타난다. inf 없거나 NaN 값인 경우 모두다 0으로 들어간다

# 데이터 프레임의 연산
# 데이터 프레임에 연산하는 가정

# titanic age , fare  두개 열을 선택해서 데이터 프레임 만들기

titanic = sns.load_dataset('titanic')   # 샘플 데이터를 가져오는 것이다. titanic / iris / flights 가 있다.
df = titanic.loc[ : , ['age','fare']]
# print(df.head())    # 첫 5행만 
# print('\n')
# print(type(df))
# print('\n')


# addition = df+10        # 모든 원소에 숫자 10을 더하고 데이터 프레임의 크기와 모양은 변경되지 않는다.
# print(addition.head())
# print('\n')
# print(type(addition))

# print(df.tail())    # 마지막 5행 표시
# print('\n')
# print(type(df))
# print('\n')

# addition = df + 10
# print(addition.tail())
# print('\n')

# ad_df= addition - df
# print(ad_df.tail())
