from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


def build_clinical_vector_store():
    loader = PyPDFDirectoryLoader("medical_docs")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # embeddings = GoogleGenerativeAIEmbeddings(
    #     model="models/gemini-embedding-001", client_options={"api_version": "v1"}
    # )

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = FAISS.from_documents(chunks, embedding_model)
    vector_store.save_local("faiss_index_hugging_face")
    # vector_store.save_local("faiss_index")


if __name__ == "__main__":
    build_clinical_vector_store()
