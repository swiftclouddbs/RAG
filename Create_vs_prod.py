from langchain.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain.document_loaders import PyPDFLoader  # To load PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Initialize the embeddings model for vector search
# You can use OpenAIEmbeddings, NomicEmbeddings, or others
embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

# Specify the directory where the Chroma vector data will be stored
PERSIST_DIRECTORY = "./Map"  # Update with your desired directory

# Initialize the Chroma vector store
chroma_store = Chroma(
    collection_name="Old_Map", 
    embedding_function=embedding_model,
    persist_directory=PERSIST_DIRECTORY  # Specify persistent directory
)

# Load the PDF file(s)
pdf_file_path = "America.pdf"  # Replace with your PDF file path

# Use PyPDFLoader to extract the content from the PDF
loader = PyPDFLoader(pdf_file_path)
documents = loader.load()

# Split the text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
texts = text_splitter.split_documents(documents)

# Add the extracted texts (as Documents) to Chroma vector store
chroma_store.add_documents(texts)

# Persist the store to disk for future retrieval
chroma_store.persist()

print("PDF content successfully loaded into Chroma vector store.")
