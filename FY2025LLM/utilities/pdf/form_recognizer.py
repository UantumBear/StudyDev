class AzureFormRecognizerClient:
    def __init__(self, client):
        self.client = client

    def extract_layout(self, form_url: str):
        poller = self.client.begin_analyze_document_from_url("prebuilt-layout", form_url)
        return poller.result()
