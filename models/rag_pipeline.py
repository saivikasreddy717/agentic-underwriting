from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings  # or another embedding model
from langchain.chains import RetrievalQA

def setup_vector_store(documents):
    # documents: list of underwriting guideline texts
    embeddings = OpenAIEmbeddings()  # initialize your embeddings (replace if needed)
    vector_store = FAISS.from_texts(documents, embeddings)
    return vector_store

def get_guidelines(query, vector_store):
    # Create a retrieval QA chain that fetches relevant guidelines
    retrieval_chain = RetrievalQA.from_chain_type(
        llm="gpt-3.5-turbo",  # or use your access to Gemini 2.0 Flash if available
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    result = retrieval_chain.run(query)
    return result
