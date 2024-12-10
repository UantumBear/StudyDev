import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Reload the Excel file with the updated request
file_path_updated = 'data/clustered_data_groupby.xlsx'
updated_data = pd.ExcelFile(file_path_updated)

# Initialize a dictionary to store the representative coin for each sheet (cluster)
updated_representative_coins = {}

# Iterate through each sheet to perform PCA and select the representative coin
for sheet_name in updated_data.sheet_names:
    # Load the sheet into a DataFrame
    df_sheet = updated_data.parse(sheet_name)

    # Extract numerical columns for correlation analysis (exclude non-numeric like 'Date' if present)
    numeric_columns = df_sheet.select_dtypes(include=['float64', 'int64'])

    # Calculate the correlation matrix
    correlation_matrix = numeric_columns.corr()

    # Apply PCA on the correlation matrix
    pca = PCA(n_components=1)
    pca.fit(correlation_matrix)

    # Get the loadings for the first principal component
    loading_matrix = pd.DataFrame(
        pca.components_.T,
        columns=['PC1'],
        index=correlation_matrix.columns
    )

    # Identify the coin with the highest absolute contribution to PC1
    representative_coin = loading_matrix['PC1'].abs().idxmax()
    updated_representative_coins[sheet_name] = representative_coin

# Display the updated representative coins for each sheet
updated_representative_coins
print(f"updated_representative_coins: \n", updated_representative_coins)