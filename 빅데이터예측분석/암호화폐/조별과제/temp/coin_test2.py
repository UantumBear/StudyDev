# api_key = "JU59ECUA5VUPYE5AFVJG5C3EIIX7K2USPW"
import requests
import pandas as pd

# API 설정
api_key = 'JU59ECUA5VUPYE5AFVJG5C3EIIX7K2USPW'
address = '0x00000000219ab540356cBB839Cbe05303d7705Fa'  # 예: 유명 주소 "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=desc&apikey={api_key}"

# API 요청
response = requests.get(url)
data = response.json()
# print(data)  # API 응답 전체 확인




if data['status'] == '1':
    # 데이터를 DataFrame으로 변환
    df = pd.DataFrame(data['result'])

    # 타임스탬프를 날짜로 변환
    df['timeStamp'] = pd.to_datetime(df['timeStamp'].astype(int), unit='s')

    # ETH 값 변환 (Wei -> ETH)
    df['value'] = df['value'].astype(float) / (10 ** 18)

    # ETH 거래만 필터링 (value > 0)
    eth_transactions = df[df['value'] > 0].copy()

    # 주요 열 선택
    # eth_transactions = eth_transactions[['timeStamp', 'from', 'to', 'value', 'gas', 'gasPrice']]

    # 결과 확인
    print(eth_transactions.head())

    # CSV로 저장
    eth_transactions.to_csv("eth_0x00000000219ab540356cBB839Cbe05303d7705Fa_DESC.csv", index=False, encoding="utf-8-sig")
    print("ETH 거래 데이터가 csv'로 저장되었습니다.")
else:
    print(f"API 호출 실패: {data['message']}")