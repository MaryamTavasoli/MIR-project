from index_reader import Index_reader
from indexes_enum import Indexes, Index_types
import json

class Metadata_index:
    def __init__(self, path='index/'):
        """
        Initializes the Metadata_index.

        Parameters
        ----------
        path : str
            The path to the indexes.
        """

        #TODO
        self.documents = self.read_documents(path)
        self.metadata_index = self.create_metadata_index()
    def read_documents(self,path):
        """
        Reads the documents.

        """

        #TODO
        documents_index = Index_reader(path, Indexes.DOCUMENTS)
        print(documents_index.index)
        return documents_index.index['documents']

    def create_metadata_index(self):
        """
        Creates the metadata index.
        """
        metadata_index = {}
        metadata_index['averge_document_length'] = {
            'stars': self.get_average_document_field_length('stars'),
            'genres': self.get_average_document_field_length('genres'),
            'summaries': self.get_average_document_field_length('summaries')
        }
        metadata_index['document_count'] = len(self.documents)

        return metadata_index

    def get_average_document_field_length(self,where):
        """
        Returns the sum of the field lengths of all documents in the index.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.
        """

        #TODO
        document_lengths_index = DocumentLengthsIndex()
        total_length = sum(doc_length_field for doc_length_field in document_lengths_index.get_documents_length(where).values())
        return total_length / len(self.documents) if len(self.documents) > 0 else 0

    def store_metadata_index(self, path):
        """
        Stores the metadata index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        """
        path =  path + Indexes.DOCUMENTS.value + '_' + Index_types.METADATA.value + '_index.json'
        with open(path, 'w') as file:
            json.dump(self.metadata_index, file, indent=4)



if __name__ == "__main__":
    meta_index = Metadata_index()
    meta_index.store_metadata_index('index/')