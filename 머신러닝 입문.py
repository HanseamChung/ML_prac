import numpy as np

x = np.array([[1,2,3], [4,5,6]])
print('x:\n{}'.format(x))
from scipy import sparse

#대각선 원소는 1이고 나머지는 0인 2차원 NumPy배열
eye = np.eye(4)
print('NumPy 배열 : \n{}'.format(eye))

#NumPy 배열을 CSR 포맷의 SCiPy 희소 행렬로 변환
#0이 아닌 원소만 저장
sparse_matrix = sparse.csr_matrix(eye)
print("SciPy의 CSR 행렬 : \n{}".format(sparse_matrix))

data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO 표현 :\n{}".format(eye_coo))

%matplotlib inline
import matplotlib.pyplot as plt

#-10에서 10까지 100개의 간격으로 나뉜 배열을 생성합니다.
x = np.linspace(-10, 10, 100)
#사인(sin) 함수를 사용하여 y배열을 생성합니다
y = np.sin(x)
#플롯(plot) 함수는 한 배열의 값을 다른 배열ㅇ 대응해서 선 그래프를 그립니다.
plt.plot(x, y, marker='x')

import pandas as pd

#회원정보가 들어간 간단한 데이터셋을 생성합니다
data = { 'Name' : ['John','Anna', 'peter', 'Linda'],
        'Location' : ['New York', 'Paris', 'Berlin', 'London'],
        'Age' : [24,13,53,33]}

data_pandas = pd.DataFrame(data)
#Ipython.display는 주피터 노트북에ㅓ Dataframe을 미려하게 출력해준다
display(data_pandas)

# Age 열의 값이 30 이상인 모든 행을 선택
display(data_pandas[data_pandas.Age >30])


##시스템 버전 확인하기
import sys
print('Python 버전 : {}'.format(sys.version))
import pandas as pd
print('pandas 버전 : {}'.format(pd.__version__))

import matplotlib
print('matplotlib 버전 : {}'.format(matplotlib.__version__))

import numpy as np
print('numpy 버전 : {}'.format(np.__version__))

import scipy as sp
print('scipy 버전 : {}'.format(sp.__version__))


@@@@@패키지 스파이더에 설치하는 방법
import Ipython

##데이터 적재
from sklearn.datasets import load_iris
iris_dataset = load_iris

@@@@@@@@@@@@@@@@@@@@@@@keys함수가 없다고함
print('iris_dataset의 키 : \n{}'.format(iris_dataset.keys()))

print(iris_dataset['DESCR'][:193] + '\n...')

