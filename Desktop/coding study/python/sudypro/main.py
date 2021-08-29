# Dictionary 를 할용하여 DataFrame 생성
import pandas as pd
import numpy as np

data = {
    'country': ['china', 'japan', 'korea', 'usa'],
    'gdp': [1409250000, 516700000, 169320000, 2041280000],
    'population': [141500, 12718, 5180, 32676]
}

country = pd.DataFrame(data)    # 데이터 프레임에 넣기 여기까지는 인덱스가 숫자다
country = country.set_index('country')  # 인덱스를 영어로 셋팅
# country 는 index를 세팅하겠다. 이때, 컨트리를 인덱스화를 시킨다.

# 딕셔너리 는 data를 중괄호로 표기한다 딕셔너리에서 시리즈를 만들수 있다.
# 시리즈는 ([ 바로 만들 수 있다
# 데이터 프레임을 만들 수 있다.

print(country.shape)        # 데이터의 벨류만을 모양으로 의미한다.
print(country.size)
print(country.ndim)
print(country.values)
print(country)

# DataFrame 의 index 와 column에 이름 지정 가능하다

country.index.name = "Country"
country.columns.name =  'info'
print(country.index)
print(country.columns)
print(country)

# Series 도 넘파이 처럼 연산자 사용 가능
print('-')
# 이번에는 데이터 프레임에 새로운 데이터 추가하기 list, dict

df = pd.DataFrame(columns= ['이름','나이','주소'])    # 데이터 프레임 뚜껑 만들기
df.loc[0] = ['길동', '26', '서울']  # list 로 추가히기   # 리스트 사용법
df.loc[1] = {'이름':'철수', '나이':25, '주소':'인천'}     # 딕셔너리 사용
df.loc[1, '이름'] = '영회' # 명시적 인덱스 사용해서 수정

#  NaN값으로 초기화 한 새로운 컬럼 추가
df['전화번호'] = np.nan # 자리를 비워두고 시작한다 not a number 숫자가 아니다를 의미한다.
df.loc[0, '전화번호'] = '01012341234' # 저장한당

# DataFrame 에서 컬럼 삭제 후 원본 변경
df.drop('전화번호', axis = 1, inplace = True)       # 행에 방향으로 삭제할 것인가 열의 방향으로 삭제할 것인가가 중요하다.
# axis = 1 : 열 방향 / axis = 0: 행방향
#  inplace 트루는 원본 변경 inplace 펄스는 원본 변경 x # 원본데이터를 꼭 따로 저장해놓고 사용해야 한다.
