""" 문서 처리 기능을 수행하는 파이프라인

"""
class DocumentPipeline:
    def __init__(self, form_recognizer, table_parser, cleaner, blob_client, embedder):
        self.form_recognizer = form_recognizer
        self.table_parser = table_parser
        self.cleaner = cleaner
        self.blob_client = blob_client
        self.embedder = embedder

    def run(self, file_url: str, prcProgId: str):
        layout = self.form_recognizer.extract_layout(file_url)
        tables_raw = self.table_parser.parse_tables(layout)

        for table in tables_raw:
            df = self.cleaner.to_clean_dataframe(table["matrix"])
            texts = df.apply(lambda row: " | ".join(row.values.astype(str)), axis=1).tolist()

            # 업로드 및 임베딩
            sas_url = self.blob_client.upload_file(file_name="uploaded.pdf", bytes_data=b"", prcProgId=prcProgId)
            self.embedder.insert_embeddings(texts, metadata={"sas_url": sas_url, "prc_prog_id": prcProgId})
