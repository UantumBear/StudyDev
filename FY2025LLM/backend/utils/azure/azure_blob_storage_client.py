"""
Azure Blob Storage 업로드 및 SAS URL 생성
"""
# pip install azure-storage-blob
from azure.storage.blob import BlobServiceClient, ContentSettings, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
import os

class AzureBlobStorageClient:
    def __init__(self, container_name=None):  # 컨테이너는 목적에 따라 유동적으로 입력받아 지정.

        self.account_name = os.getenv("AZURE_BLOB_ACCOUNT_NAME")
        self.account_key = os.getenv("AZURE_BLOB_KEY1")
        self.account_url = os.getenv("AZURE_BLOB_URL1")
        self.container_name = container_name or os.getenv("AZURE_BLOB_CONTAINER") # 만일 입력받지 않는다면 기본 컨테이너 경로를 사용

        if not all([self.account_name, self.account_key, self.account_url, self.container_name]):
            # 누락된 키 확인
            missing_keys = []
            if not self.account_name:
                missing_keys.append("AZURE_BLOB_ACCOUNT_NAME")
            if not self.account_key:
                missing_keys.append("AZURE_BLOB_KEY1")
            if not self.account_url:
                missing_keys.append("AZURE_BLOB_URL1")
            if not self.container_name:
                missing_keys.append("AZURE_BLOB_CONTAINER")

            if missing_keys:
                for key in missing_keys:
                    print(f"[환경변수 누락] {key} is missing.")


            raise EnvironmentError("Blob 관련 환경변수가 누락되어 초기화에 실패했습니다.")

        self.client = BlobServiceClient(account_url=self.account_url, credential=self.account_key)
        self.container = self.client.get_container_client(self.container_name)

    def upload_file_and_get_sas(self, file_name, data_bytes, expires_in_hours=3):
        blob = self.container.get_blob_client(file_name)
        blob.upload_blob(
            data_bytes,
            overwrite=True,
            content_settings=ContentSettings(content_type="text/plain")
        )

        sas_token = generate_blob_sas(
            account_name=self.account_name,
            container_name=self.container_name,
            blob_name=file_name,
            account_key=self.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=expires_in_hours)
        )

        return f"{blob.url}?{sas_token}"
