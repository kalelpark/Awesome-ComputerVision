# 판다스 자료구조 -> 구조화된 데이터 형식을 제공
# Series -> index  : value -> 딕셔너리와 비슷한 구조

import pandas as pd
# from pandas.core.indexes.base import Index

# dict_data = {'a': 1, 'b': 2, 'c': 3}

# sr = pd.Series(dict_data)

# print(type(sr))   
# print('\n')
# print(sr)

# ---------- # 리스트를 벨류값으로 사용하면서 이용 가능하다. 단 이때, index가 없으므로 0부터 나타난다
# list_data = ['2019-01-02', 3.14, 'ABC', 100, True]
# sr = pd.Series(list_data)
# # print(sr)

# idx = sr.index 
# val = sr.values
# print(idx)  # 인덱스의 형태를 알려준다 Range로 
# print(val)  # 형태를 알려준다

# -- 원소의 선택 리스트 슬라이싱 기법과 비슷하게 인덱스 범위를 지정하여 원소를 선택 가능
# 정수형 인덱스는 대괄호 안에 숫자 입력, 이름 인덱스는 작은 따움표를 사용한다.

# # 튜플 인덱스 미리 지정해주기
# tup_data = ('영민', '2010-05-10', '여', True)
# sr = pd.Series(tup_data , index=['이름', '생년월일', '성별', '학생여부'])
# print(sr)

# print()
# print(sr[0])
# print(sr['이름'])   # 인덱스로 표현 가능

# print(sr[[1, 2]])
# print(sr[['이름', '생년월일']])       # 인덱스 표현 (두가지 이상 선택을 할경우 리스트안에 리스트를 넣기)

# 범위를 지정하는 방식

# print(sr[1:2])      # 숫자는 을 포함을 하지 않지만
# print()
# print(sr['생년월일':'성별'])        # 인덱스 이름은 포함한다.

#---------------- Series  끝
#---------------- DataFrame  시작
# DataFrame은 2차원 배열이다. 즉 , 여러 개의 열 벡터들이 같은 행 인덱스를 기준으로 줄지어 결합된 2차원 벡터 또는 행렬

# 데이터 프레임은 행과 열을 나타내기 위해서 두가지 종류의 인덱스 사용
# 열은 공통 속성을 갖는 일련의 데이터 를 나타내고, 행은 개별 관측 대상의 다양한 속성들의 모음인 record 를 나타낸다. # 행은 관측값을  , 열은 공통범주

# DAtaframe 만들기
# 딕셔너리의 키의 값이 열의 이름이 되고, 값에 해당하는 각 리스트가 데이터 프레임의 열이 된다.

# dict_data = {'co':[1, 2, 3], 'c1':[4, 5, 6], 'c2':[7, 8, 9], 'c3':[10, 11, 12], 'c4':[13, 14, 15]}
# df = pd.DataFrame(dict_data)
# print(df)

# 행 / 열 인덱스 이름 설정

# import pandas as pd

# df = pd.DataFrame([[15, '남', '덕영중'], ['20', '여', '스리대']],
#                     index = ['준서', '호연'],       # 헹 인덱스
#                     columns = ['나이', '성별', '이름'])     # 열 인덱스
# 위에는 첫번째 접근 방식이며,
# 두번째 접근 방식에는
# df.index = ['준희' , '나열']
# df.columns = ['나이2', '성별2', '이름2']
# print(df)

# df.rename(columns={'나이':'연령', '이름':'학벌'}, inplace = True)
# df.rename(index={'준서':'준테', '호연':'호긴'}, inplace= True)      # rename을 사용할 경우, 원본 객체를 변환한다는 의미에서 inplace = True 를 사용해주어야 한다.
# print(df)

# 행 / 열 삭제 
# drop 기존 객체를 변경하지 않고, 새로운 객체를 반환한다. 원본객체가 변경된다 (inplace = True 로 사용할 경우), (inplace = False 인 경우에는 원본은 유지된 채 새로운 것을 전달)


# exam_data = [[90, 98, 85, 100], [80, 89, 95, 90], [70, 95, 100, 90]]
# # or 

# df = pd.DataFrame(exam_data)
# df.index = ['서준', '우연', '인아']
# df.columns = ['수학','영어','음악','체육']

# print(df)
# print()

# df2 = df[:]     # 복제시 [:] 를 사용하여 복제하기
# df2.drop('우연', inplace=True)

# print(df2)

# print()

# df3 = df[:]
# df3.drop(['우연'], axis = 0, inplace = True)
# df3.drop(['인아'], axis = 0, inplace = True)
# print(df3)


# --- 열을 삭제하는 방법
# 반드시 축 옵션을 axis = 1로 설정해야 한다.

# exam_data = {'수학' : [90, 80, 70], '영어' : [98, 89, 95], 
#             '음악': [80, 93, 23], '과학' : [23, 32, 43]}
# df = pd.DataFrame(exam_data, index=['서준', '예준', '호준'])
# print(df)

# df4 = df.copy()
# df4.drop('수학', axis=1, inplace=True)
# print(df4)
# print()
# df5 = df.copy()
# df5.drop(['영어', '과학'], axis=1, inplace=True)
# print(df5)


# 행 선택은 정수형은 범위로 끝에 제외이지만 문자형은 포함이다.

# label1 = df.loc['서준'] # 인덱스 이름
# position1 = df.iloc[0] # 정수형 위치 인덱스     # 행 인덱스의 점수를 알려준다.

# print(label1)
# print()
# print(position1)

# label2 = df.loc[['서준', '예준']]
# position2 = df.iloc[[0, 2]]       # 여러명일 경우 괄호 두개
# print(label2)
# print(position2)

# 여러행 동시에 호출하는 방법

# 행 인덱스의 범위를 지정하여 여러 개의 행 동시에 선택
# label3 = df.loc['서준': '호준']
# position3  = df.iloc[0:2]
# print(label3)
# print(position3)

# 행을 여러개 동시에 선택하는 방법

# 열  선택 하기
# import pandas as pd

exam_data = {'이름': ['서준', '우현', '민수'],
             '수학': [90, 80, 50],
             '영어': [80, 30, 40],
             '사회': [90, 20, 30],
             '과학': [90, 30, 50]}

df = pd.DataFrame(exam_data)
# print(df)

# math1 = df['수학']
# print()
# print(math1)
# print()
# print(df.영어)

# 열 선택에서 호출하기

# 두개이상의 열 추출하기

# music_gym = df[['영어', '과학']]
# print(music_gym)

# 범위 슬라이싱 고급 활용 
# df.iloc[시작 : 끝 : 간격]     # 만약 열을 선택하고 싶다면 괄호를 두개 사용

# 원소 선택 

df.set_index ('이름', inplace=True)         # set_index 는 새로운 행 인덱스로 지정한다
print(df)

# # 데이터 프레임에서 원소 한개를 출력하는 방식
print()
# a = df.iloc[0, 2]
# print(a)

# # 두개 이상 선택하는 방법
# c= df.loc['서준' , ['영어', '사회']]
# c =df.iloc[0, [1,3]]
# print(c)

# 두개 이상
# c = df.loc[['서준','우현'], ['영어', '수학']] # 앞부분 가로 뒤 세로
# print(c)
# print()
# c = df.iloc[[0 , 2] , [1, 3]]
# print(c)
# c= df.iloc[0:2 , 2:]
# print(c)

