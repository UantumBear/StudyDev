"""
 매트릭스를 DataFrame으로 바꾸고 _concat 병합, NaN 제거

"""
class TableCleaner:
    def to_clean_dataframe(self, matrix: List[List[str]]) -> pd.DataFrame:
        df = pd.DataFrame(matrix[1:], columns=matrix[0])
        df.replace("...", np.nan, inplace=True)
        df.dropna(how="all", inplace=True)
        df.dropna(how="all", axis=1, inplace=True)
        df.replace("\n", "", regex=True, inplace=True)
        return self._merge_concat_columns(df)

    def _merge_concat_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        col_to_length = defaultdict(int)
        for col in df.columns:
            if "_concat" in col:
                col_to_length[col.replace("_concat", "")] += 1

        for col, count in col_to_length.items():
            concat_col = col + "_concat"
            if count == 1:
                df.rename(columns={concat_col: col}, inplace=True)
            else:
                df[col] = pd.Series(df[concat_col].fillna('').values.tolist()).str.join("")
                del df[concat_col]

        return df
