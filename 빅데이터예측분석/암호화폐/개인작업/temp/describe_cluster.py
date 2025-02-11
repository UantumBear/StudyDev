import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# 한글 폰트 경로 지정
font_path = 'fonts/malgun.ttf'
font_prop = fm.FontProperties(fname=font_path, size=8) # 폰트 프로퍼티 설정
# matplotlib의 rcParams 설정을 통해 전역적으로 한글 폰트 적용
plt.rcParams['font.family'] = font_prop.get_name()

# Excel 파일 경로
file_path = 'data/clustered_data_groupby.xlsx'
# 색상 팔레트 설정
colors = ['red', 'blue', 'green', 'purple', 'orange']
# Excel 파일 읽기
with pd.ExcelFile(file_path) as xls:
    sheet_names = xls.sheet_names  # 시트 이름 목록
    std_data = []  # 표준편차 데이터를 저장할 리스트

    # 각 시트의 데이터에 대한 통계적 요약 수행
    for sheet_index, sheet_name in enumerate(sheet_names):
        # 시트 데이터 불러오기
        data = pd.read_excel(xls, sheet_name=sheet_name)

        # 통계적 요약 계산
        summary = data.describe()  # 평균, 표준편차, 최소/최대값, 사분위수 포함
        median = data.median()  # 중앙값 계산

        # 중앙값을 요약 데이터에 추가
        summary.loc['median'] = median

        # 결과 출력
        print(f"Statistics for {sheet_name}:")
        print(summary)
        print("\n")  # 각 시트의 결과를 구분하기 위해 빈 줄 추가

        # 표준편차 데이터 수집
        std_data.append(summary.loc['std'])

# 그래프 그리기
plt.figure(figsize=(10, 6))
for index, std in enumerate(std_data):
    plt.scatter(std.index, std.values, color=colors[index % len(colors)], label=f'Cluster {index}', s=100)

plt.title('클러스터별 표준편차(Standard Deviation by Cluster)')
plt.xlabel('암호화폐(Crypto)')
plt.ylabel('표준편차(Standard Deviation)')
plt.legend(title='Cluster')
plt.xticks(rotation=90)  # Crypto 이름이 겹치지 않도록 회전
plt.tick_params(axis="x", labelsize=7)
plt.grid(True)
# 그래프를 파일로 저장
plt.savefig('data/클러스터별표준편차.png', format='png', dpi=300)