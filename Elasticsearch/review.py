# pdf_parser


# convert : pdf to txt
from azure.ai.formrecognizer import DocumentAnaysisClient
from azure.core.credentials import AzureKeyCredential

documentAnaysis = DocumentAnaysisClient(
    endpoint=self.form_recognizer_endpoint,
    credential=AzureKeyCredential(slef.form_recognizer_key)
)

poller = documentAnaysis.begin_anayze_document_from_url("prebuilt-layout", formUrl)
layout = poller.result()



