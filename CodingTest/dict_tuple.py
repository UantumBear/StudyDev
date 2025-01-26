import sys

# 입력받기
n, m = map(int, input().split(' '))  # N,M
# print(f"n: {n}, m: {m}")
weight_no_list = list(map(int, input().split(' ')))  # W1, W2, ... , WN
# 친분 관계 입력 받기
pairs = []  # [ (1,2) , (2,4) , ... ] 이런 형식
for i in range(m):
    pair = tuple(map(int, input().split(' ')))  # ex) 2 3
    pairs.append(pair)
# print(f"pairs: {pairs}")

# 이제 입력은 다 받았으니 데이터 처리
pairs_dict = {}
for a, b in pairs:
    # a를 key, b를 val로
    # 키가 없는 경우 먼저 키를 생성
    if a not in pairs_dict:
        pairs_dict[a] = []
    pairs_dict[a].append(b)
    if b not in pairs_dict:
        pairs_dict[b] = []
    pairs_dict[b].append(a)


class Member:
    def __init__(self, no, w, flist=[]):
        self.no = no  # 본인의 회원 넘버
        self.w = w  # 본인이 칠 수 있는 최고 중량
        self.flist = flist  # 본인의 친구 리스트
        self.top = False  # 본인이 최고라고 생각하면 true


members = []
# 회원 선언
for i in range(n):
    no = i + 1
    if no in pairs_dict:  # 친구가 있다면 리스트를 넣고
        member = Member(no=no, w=weight_no_list[i], flist=pairs_dict[no])
        # print(f"pairs_dict[{no}] : {pairs_dict[no]}")
    else:  # 없다면 넣을 필요 없음
        member = Member(no=no, w=weight_no_list[i])
    members.append(member)
# 회원끼리 중량 비교
for member in members:
    # 본인의 중량과 친구의 중량을 비교
    for friend_no in member.flist:  # friend_no = 2, 4, ...
        friend = members[friend_no - 1]
        # 한번이라도 친구보다 중량이 낮거나 같으면 최고가 아님
        if member.w <= friend.w:
            member.top = False
            break
    else:
        member.top = True

# 헬스장에서 본인이 최고라고 생각하는 회원의 수
iam_top = 0
for member in members:
    if member.top == True:
        iam_top += 1

print(iam_top)
