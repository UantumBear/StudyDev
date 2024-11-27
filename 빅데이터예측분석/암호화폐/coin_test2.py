# api_key = "JU59ECUA5VUPYE5AFVJG5C3EIIX7K2USPW"
import requests
import pandas as pd

# API 설정
api_key = 'JU59ECUA5VUPYE5AFVJG5C3EIIX7K2USPW'
address = '0xa7efae728d2936e78bda97dc267687568dd593f3'  # 예: 유명 주소 "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={api_key}"

# API 요청
response = requests.get(url)
data = response.json()
# print(data)  # API 응답 전체 확인


# 데이터 확인 및 처리
if data['status'] == '1':
    transactions = data['result']

    # 트랜잭션 데이터가 없는 경우
    if len(transactions) == 0:
        print("트랜잭션 데이터가 없습니다.")
    else:
        # DataFrame으로 변환
        df = pd.DataFrame(transactions)

        # DataFrame의 첫 몇 줄과 열 이름 확인 (디버깅용)
        print(df.head())
        print(df.columns)

        # 타임스탬프를 날짜로 변환
        df['timeStamp'] = pd.to_numeric(df['timeStamp'])
        df['timeStamp'] = pd.to_datetime(df['timeStamp'], unit='s')

        # 전체 데이터의 시간 범위 확인
        print("최소 타임스탬프:", df['timeStamp'].min())
        print("최대 타임스탬프:", df['timeStamp'].max())

        # 필터링 없이 모든 데이터 저장
        df.to_csv("ethereum_all_transactions.csv", index=False, encoding="utf-8-sig")
        print("전체 트랜잭션 데이터가 'ethereum_all_transactions.csv'로 저장되었습니다.")

else:
    print(f"API 호출 실패: {data['message']}")