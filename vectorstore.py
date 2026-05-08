from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from knowledgebase import DOCUMENTS

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# from langchain_openai import OpenAIEmbeddings

# # Ensure your OPENAI_API_KEY is set in your environment variables
# embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = FAISS.from_texts(DOCUMENTS, embedding_model)

def retrieve_context(query: str, k: int = 3) -> str:
    docs = vector_store.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs])
