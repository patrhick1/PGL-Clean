# app/google_docs_service.py

import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

SCOPES = [
    'https://www.googleapis.com/auth/documents',
    'https://www.googleapis.com/auth/drive'
]
SERVICE_ACCOUNT_FILE = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
FOLDER_ID = os.getenv('GOOGLE_PODCAST_INFO_FOLDER_ID')

class GoogleDocsService:
    def __init__(self):
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        self.docs_service = build('docs', 'v1', credentials=credentials)
        self.drive_service = build('drive', 'v3', credentials=credentials)

    def get_document_content(self, document_id):
        doc = self.docs_service.documents().get(documentId=document_id).execute()
        content = self._read_structural_elements(doc.get('body').get('content'))
        return content

    def _read_structural_elements(self, elements):
        text = ''
        for element in elements:
            if 'paragraph' in element:
                text += self._read_paragraph_elements([element])
            elif 'table' in element:
                text += self._read_table_elements([element])
            elif 'tableOfContents' in element:
                text += self._read_tableOfContents_elements([element])
        return text

    def _read_paragraph_elements(self, elements):
        text = ''
        for element in elements:
            if 'paragraph' in element:
                for run in element.get('paragraph').get('elements'):
                    if 'textRun' in run:
                        text += run.get('textRun').get('content')
        return text

    def _read_table_elements(self, elements):
        text = ''
        for element in elements:
            if 'table' in element:
                for row in element.get('table').get('tableRows', []):
                    for cell in row.get('tableCells', []):
                        text += self._read_paragraph_elements(cell.get('content',[]))

        return text
    
    def _read_tableOfContents_elements(self, elements):
        text = ''
        for element in elements:
            if 'tableOfContents' in element:
                toc = element.get('tableOfContent')
                text += self._read_paragraph_elements(toc.get('content', []))

        return text    

    def create_document_without_content(self, title, folder_id):
        # Create the Google Doc
        body = {'title': title}
        doc = self.docs_service.documents().create(body=body).execute()
        doc_id = doc.get('documentId')

        # Move the doc to the specified folder
        self.drive_service.files().update(
            fileId=doc_id,
            addParents=folder_id,
            fields='id, parents'
        ).execute()
        return doc_id


    def create_document(self, title, content, folder_id=None):
        # Create a new Google Doc
        body = {'title': title}
        doc = self.docs_service.documents().create(body=body).execute()
        document_id = doc.get('documentId')

        # Insert content into the document
        requests = []
        requests.append({
            'insertText': {
                'location': {
                    'index': 1,
                },
                'text': content
            }
        })
        self.docs_service.documents().batchUpdate(
            documentId=document_id, body={'requests': requests}).execute()

        # Move the document to the desired folder
        if folder_id:
            file = self.drive_service.files().update(
                fileId=document_id,
                addParents=folder_id,
                fields='id, parents'
            ).execute()

        # Return the document link
        return f'https://docs.google.com/document/d/{document_id}/edit'

    def share_document(self, document_id):
        drive_service = self.drive_service
        
        permission = {
            'type': 'anyone',
            'role': 'writer',
            'allowFileDiscovery': False
        }
        
        drive_service.permissions().create(
            fileId=document_id,
            body= permission,
            fields='id'
        ).execute()
    
    def append_to_document(self, document_id, content):
        requests = [
            {
                'insertText': {
                    'location': {'index': 1},
                    'text': content
                }
            }
        ]
        result = self.docs_service.documents().batchUpdate(
            documentId=document_id, body={'requests': requests}).execute()
        return result
    
    def check_file_exists(self, file_name):
        query = f"name = '{file_name}' and trashed = false"
        results = self.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        files = results.get('files', [])
        return len(files) > 0
    

    def check_file_exists_in_folder(self, file_name, folder_id=FOLDER_ID):
        query = f"name = '{file_name}' and '{folder_id}' in parents and trashed = false"
        results = self.drive_service.files().list(q=query, 
                                    spaces='drive', 
                                    fields='files(id, name)').execute()
        files = results.get('files', [])
        #return len(files) > 0
        if files:
            return True, files[0]['id']  # Return True and the first file's ID if found
        return False, None


if __name__ == "__main__":
    google_service = GoogleDocsService()
    document_id = "1qI7MeCuWZN1d9Jf76AAgLiUb1ODXI1GakCPSClXxz0k"
    title = "Hvac Uncenso"
    search_file = google_service.check_file_exists_in_folder(title)
    print(search_file)
    if search_file[0] == True:
        print('file is found')
        print(search_file[1])
    else:
        print('Not found')
        print(search_file)