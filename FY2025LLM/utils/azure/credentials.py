import os
# pip install azure-core
from azure.core.credentials import AzureKeyCredential

def get_azure_credential():
    return AzureKeyCredential(os.getenv("AZURE_KEY"))

def get_azure_endpoint(service: str) -> str:
    return os.getenv(f"AZURE_{service.upper()}_ENDPOINT")
