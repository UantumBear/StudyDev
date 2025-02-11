import sys

n = int(input())  # 문자열 쌍의 개수
s_list = []
t_list = []
for i in range(n):
    s, t = input().split()
    # print(f"s: {s}, t:{t}")
    s_list.append(s.upper())
    t_list.append(t.upper())
    # print(f"s_list: {s_list}, t_list:{t_list}")

# s 에 들어있는  x 혹은 X 의 index 를 기억해서, t의 해당 index 에 있는 글자를 대문자로 읽기
s_index_list = []
t_string = ""
t_char_list = []
# for s in s_list:
#     for index, char in enumerate(s):
#         if 'x' == char or 'X' == char:
#             s_index_list.append(index)
#             break
# print(s_index_list)

for rownum, s in enumerate(s_list):
    # x_index = s.index('X')
    x_index = s.find('X')
    # t_string += t_list[rownum][x_index] # 문자열 덧셈이 시간이 오래 걸린다고 함.
    t_char_list.append(t_list[rownum][x_index]) # 때문에 일단 list 로 모아준 후 아래에서 join 하여 합치기. 이렇게 하면 5초가 걸리던 문자열이 0.1초로 줄어든다.

t_string = "".join(t_char_list)
print(t_string)

