# models/rag_pipeline.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

def setup_vector_store(documents):
    # Use a high-quality free embedding model; all-mpnet-base-v2 is a good choice.
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vector_store = FAISS.from_texts(documents, embeddings)
    return vector_store

def get_guidelines(query, vector_store):
    # Use Llama 2 for text generation.
    # Here we're using the Llama 2 7B Chat model available on HuggingFace.
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Change this if you prefer a different variant.
    
    # Initialize the tokenizer and model.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    # Create a text-generation pipeline.
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7,
    )
    
    # Wrap the pipeline with LangChain’s LLM interface.
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
    # Create the retrieval chain using the Llama 2 powered LLM.
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    
    result = retrieval_chain.run(query)
    return result
