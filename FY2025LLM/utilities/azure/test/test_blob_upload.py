"""
azure 의 기본 기능들을 테스트하기 위한 테스트 스크립트
@ utils/azure/test/test_blob_upload.py
"""
import os
from utilities.azure.azure_blob_storage_client import AzureBlobStorageClient
from config import conf
from dotenv import load_dotenv

# 환경변수 불러오기
project_root_path = conf.PROJECT_ROOT_DIRECTORY
print(project_root_path)

env_path = os.path.join(conf.PROJECT_ROOT_DIRECTORY, "config", ".env")
load_dotenv(dotenv_path=env_path)

print(env_path)

account_name = os.getenv("AZURE_BLOB_ACCOUNT_NAME")
account_key = os.getenv("AZURE_BLOB_KEY1")
account_url = os.getenv("AZURE_BLOB_URL1")
print(account_name)

def test_blob_upload():
    ## 파일 가져오기
    local_path = os.path.join(project_root_path, "data", "test", "test.txt")
    # 파일 내용 읽기
    with open(local_path, "rb") as f:
        data_bytes = f.read()

    # Azure Blob에 저장할 파일 명
    blob_file_name = "test.txt"

    client = AzureBlobStorageClient(container_name="test")
    sas_url = client.upload_file_and_get_sas(file_name=blob_file_name, data_bytes=data_bytes)
    print("업로드 성공! SAS URL:")
    print(sas_url)

if __name__ == "__main__":
    test_blob_upload()

# $env:PYTHONPATH = "C:\Users\litl\PycharmProjects\gitProject\StudyDev\FY2025LLM"
# python utilities/azure/test/test_blob_upload.py