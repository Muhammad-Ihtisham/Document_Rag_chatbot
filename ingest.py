from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import EMBEDDING_MODEL, FAISS_INDEX_PATH  #  Import config values

def ingest_pdf():
    #  Load and split PDF
    loader = PyPDFLoader("data/National AI Policy Consultation Draft V1.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    #  Load embeddings using model from config
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    #  Store embeddings in local FAISS index
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)

    #  Save FAISS index to path from config
    vectorstore.save_local(FAISS_INDEX_PATH)

    print(f"Ingested {len(docs)} chunks into FAISS index and saved to '{FAISS_INDEX_PATH}'.")

if __name__ == "__main__":
    ingest_pdf()
