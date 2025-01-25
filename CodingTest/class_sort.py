## GPT 식 숫자 정렬 (Python Version 정렬 1.11 > 1.2)

import sys

class Version:
    def __init__(self, number_str):
        self.number_str = number_str
        self.int_str = 0
        self.float_str = 0
        self.order = 0

        if '.' in number_str:
            temp = number_str.split('.')
            self.int_str = int(temp[0])
            self.float_str = int(temp[1])
        else:
            self.int_str = int(number_str)

# 입력 받기
n = input()
# print(f"N: {n}")
inputs = []
for i in range(int(n)):
    inputs.append(input())
# print(f"inputs: {inputs}")

# 입력 받은 수를 클래스로 생성
versions = []
for number_str in inputs:
    versions.append(Version(number_str))

# 정렬
versions.sort(key=lambda v: (v.int_str, v.float_str))

# 정렬 결과 출력
for version in versions:
    print(version.number_str)

""" 개념
lambda 란?
lambda 란 익명 함수를 만들 때 사용하는 Python 키워드이다.
이름이 없는 간단한 함수를 만들 때 사용한다.

lambda 의 구조는 
lambda 매개변수:반환값  이다

예를 들어
def add(x,y):
    return x+y  를 사용하고 싶다면
add_lambda = lambda x,y : x+y  와 같이 정의할 수 있다.

labda 는 보통 sort, map, filter 와 같은 함수에 값을 전달하기 위한 용도로 종종 사용된다.
ex) sort
numbers = [1,5,4,3]
numbers.sort(key=lambda x:x)  # x 값을 기준으로 정렬한다. 
== 매개변수가 x, x가 반환값, 즉 이 반환값 x 를 sort 함수에 전달하는 것이다. (sort 의 반환값이 아님, 익명함수의 반환값을 말한다)
numbers.sort(key=lambda x:-x) # -x 값을 기준으로 정렬한다. 즉 sort에는 -x 값을 넣어 정렬하는 것이다.

이런 형태로
versions.sort(key=lambda v:(v.int_str, v.float_str)) 을 (int.str 을 기준으로 먼저 정렬하고, 같은 경우 float_str 을 기준으로 정렬)
하는 것이다.

그런데 v = versions 의 요소인 Version 이라고 정의한 적이 없는데 어떻게 작동할까?
이것은 sort 함수가 자동으로 리스트의 요소들에 순서대로 접근하기 때문이다.




"""


