""" 배낭에 금속 담기 """

import sys

W, N = map(int, input().split())  # W:배낭이 담을 수 있는 최대 무게, N:귀금속 종류 갯수

prices = []
for i in range(N):  # M: 금속의 무게, P: 무게당 가격 => M*P = 총 가격
    no = i + 1
    M_P = tuple(map(int, input().split()))
    prices.append(M_P)

# 문제: 배낭에 담을 수 있는 가장 "비싼 가격"을 출력하라.

"""
만약 배낭은 100kg 까지 수용 가능, 금속은 2종류
첫번째 금속의 무게는 90kg 무게 당 가격은 1
두번째 금속의 무게는 70kg 무게 당 가격은 2 라고 하면

일단 최대한 채울 수 있는 만큼 비싼 금속으로 70kg 채우고, 나머지를 싼 금속으로
=> 70*2 = 140 + (100-70)*1 = 30 => 170
"""

# [ (40, 1), (70,2), (50,1) ]

# Step. 가장 비싼 금속 찾기 (튜플의 2번째 수가 커야하며, 동일한 경우에는 일단 생각해보기)
high_sorted = sorted(prices, key=lambda x: x[1], reverse=True)
# print(high_sorted)

# Step. 가장 비싼 금속을 최대한 배낭에 쑤셔 넣기 == 최대한의 무게만큼 담기
space_W = W  # 더 담을 수 있는 무게
total_price = 0
for metal in high_sorted:  # 금속들을 순회
    m, p = metal  # 튜플 언패킹 ... ! 편하다!

    if m >= space_W:  # 금속의 무게가 더 담을 수 있는 무게와 같거나 초과한다면 그냥 그만큼 다 넣기
        cur_price = space_W * p
        total_price += cur_price
        space_W = 0
        break
    # 다 넣어도 초과하지 않는 다면 계속 넣기
    cur_price = m * p
    total_price += cur_price
    space_W = space_W - m

print(total_price)


