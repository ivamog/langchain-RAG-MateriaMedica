# Description: This script is used to generate the data store for the RAG model.


from langchain_community.document_loaders import DirectoryLoader	# For loading documents from a directory
# from langchain.document_loaders import PDFLoader		# For loading PDFs	
from langchain.schema import Document
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings	# For HuggingFace Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


import os
import shutil

CHROMA_PATH = "chroma"	# Path to the Chroma directory

DATA_PATH = "Data"	# Path to the data directory

def main():
	generate_data_store()

def generate_data_store():
	# Load the documents from the directory
	documents = load_documents()
	# Load the embeddings
	embeddings = split_text(documents)
	# print("Embeddings ", embeddings)
	# Create the Chroma data store
	save_to_chroma(embeddings)

	# Alternative solution
	# embeddings = OpenAIEmbeddings("OPENAI_API_KEY")
	# Create the Chroma data store
	# chroma = Chroma(embeddings, CHROMA_PATH)
	# Index the documents
	# chroma.index(documents)
	# Save the data store
	# chroma.save()


def load_documents():
	# Load the documents from the directory		
	loader = DirectoryLoader(DATA_PATH, glob= "*.md") # Load only .txt files
	#loader = DirectoryLoader(DATA_PATH, glob=[".pdf"]) # Load only .pdf files
	print("Data Path ", DATA_PATH)
	
	# loader = PDFLoader(DATA_PATH) # Load PDFs		
	#loader = DirectoryLoader(DATA_PATH, glob=[".pdf"]) # Load only .md marked downfiles
	documents = loader.load()
	return documents	

def split_text(documents: list[Document]):
	text_splitter = RecursiveCharacterTextSplitter(	chunk_size=2000,chunk_overlap=500, 
		length_function = len, 
		add_start_index = True,
		)
	
	MyDocuments = load_documents()
	# Split text into chunks
	# Use sliding window chunking to ensure continuity between topics
	chunks = text_splitter.split_documents(MyDocuments)
	print(f"Split {len(MyDocuments)} MyDocuments into {len(chunks)} chunks")	

	PrintDocument = chunks[1]
	print(PrintDocument.page_content)	
	print(PrintDocument.metadata)

	return chunks

# Define embedding function
def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def save_to_chroma(embeddings: list[Document]):
    # Clear out the database first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    os.makedirs(CHROMA_PATH)

    # Initialize Chroma with embedding function
    embedding_function = get_embedding_function()

    # Add metadata (Example: document source, title, etc.)
    documents_with_metadata = [
        Document(page_content=doc.page_content, metadata={"source": f"doc_{i}", **doc.metadata})
        for i, doc in enumerate(embeddings)
    ]

    # Create the Chroma database and save the data
    db = Chroma.from_documents(embeddings, embedding_function, persist_directory=CHROMA_PATH)
		
    print(f"Saved {len(documents_with_metadata)} ebmeddings to {CHROMA_PATH}.")


if __name__ == "__main__":
	main()	
