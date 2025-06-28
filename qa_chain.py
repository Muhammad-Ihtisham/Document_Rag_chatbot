from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from config import EMBEDDING_MODEL, FAISS_INDEX_PATH  #  import from config.py

def get_qa_chain():
    #  Load embeddings with model name from config
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    #  Load FAISS index from path in config
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    #  Load Ollama model (must be running)
    llm = ChatOllama(model="deepseek-r1:1.5b", temperature=0.5)

    #  Build QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    return qa_chain


