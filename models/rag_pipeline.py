import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

def setup_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vector_store = FAISS.from_texts(documents, embeddings)
    return vector_store

def get_guidelines(query, vector_store):
    model_name = "EleutherAI/gpt-j-6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create an offload folder in the current working directory.
    offload_folder = os.path.join(os.getcwd(), "offload")
    os.makedirs(offload_folder, exist_ok=True)
    
    if torch.cuda.is_available():
        # Use 8-bit quantization if CUDA is available.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
            offload_folder=offload_folder  # Specify the offload folder here.
        )
    else:
        # Load on CPU without offloading weights
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",  # Use "cpu" device_map to avoid offloading to disk.
            torch_dtype=torch.float32
        )
    
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7,
    )
    
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    result = retrieval_chain.run(query)
    return result
