# pip install azure-ai-formrecognizer
from azure.ai.formrecognizer import DocumentAnalysisClient
from .credentials import get_azure_credential, get_azure_endpoint

class AzureFormRecognizerClient:
    def __init__(self):
        endpoint = get_azure_endpoint("form_recognizer")
        credential = get_azure_credential()
        self.client = DocumentAnalysisClient(endpoint, credential)

    def analyze_layout(self, form_url: str):
        poller = self.client.begin_analyze_document_from_url("prebuilt-layout", form_url)
        return poller.result()
