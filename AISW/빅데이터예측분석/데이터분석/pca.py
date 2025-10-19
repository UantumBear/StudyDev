import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 예제 데이터셋 로드 (USArrests.csv 파일이 있는 경우)
# 주 이름을 인덱스로 설정
USArrests = pd.read_csv('usarrests.csv', index_col=0)

# 데이터 표준화
scaler = StandardScaler()
USArrests_std = scaler.fit_transform(USArrests)

# PCA 모델 생성
pca = PCA(n_components=4)
USArrests_PC = pca.fit_transform(USArrests_std)

# Biplot 그리기
plt.figure(figsize=(10, 8))

# 데이터 포인트 (주)
plt.scatter(USArrests_PC[:, 0], USArrests_PC[:, 1], color='gray')
for i, state in enumerate(USArrests.index):
    plt.text(USArrests_PC[i, 0], USArrests_PC[i, 1], state, ha='right', color='black', fontsize=9)

# 변수 로딩 벡터 (PC 로딩)
loadings = pca.components_.T
for i, var in enumerate(USArrests.columns):
    plt.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3, color='red', alpha=0.5, head_width=0.1)
    plt.text(loadings[i, 0] * 3.2, loadings[i, 1] * 3.2, var, color='red', ha='center', va='center', fontsize=12)

# 축과 제목 설정
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} 설명력)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} 설명력)")
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.title("PCA Biplot")
plt.grid()

plt.show()
