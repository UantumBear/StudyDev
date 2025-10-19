import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

""" 이것은 상관관계 분석 후 PCA 를 적용한 코드, 이렇게 해야 팀원이 구했던 대표 코인이 나온다.

"""

file_path_updated = 'data/clustered_data_groupby.xlsx'
updated_data = pd.ExcelFile(file_path_updated)


updated_representative_coins = {}


for sheet_name in updated_data.sheet_names:
    df_sheet = updated_data.parse(sheet_name)
    numeric_columns = df_sheet.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_columns.corr()

    pca = PCA(n_components=1)
    pca.fit(correlation_matrix)

    loading_matrix = pd.DataFrame(
        pca.components_.T,
        columns=['PC1'],
        index=correlation_matrix.columns
    )

    representative_coin = loading_matrix['PC1'].abs().idxmax()
    updated_representative_coins[sheet_name] = representative_coin

updated_representative_coins
print(f"updated_representative_coins: \n", updated_representative_coins)