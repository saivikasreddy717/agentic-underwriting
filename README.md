# Agentic Underwriting: AI-Powered Loan & Compliance Workflow

This repository implements an AI-powered underwriting system that automates loan compliance checks. It leverages state-of-the-art NLP techniques to analyze loan applications, verify documents, and enforce policy checks using a multi-agent approach. The project integrates FinBERT for financial text classification, a Retrieval-Augmented Generation (RAG) pipeline powered by GPT-J-6B for generative reasoning, and an interactive Streamlit interface.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Setup](#step-by-step-setup)
    - [Clone the Repository](#1-clone-the-repository)
    - [Create and Activate a Virtual Environment](#2-create-and-activate-a-virtual-environment)
    - [Install Dependencies](#3-install-dependencies)
4. [Running the Application](#running-the-application)
5. [Code Overview](#code-overview)
6. [Troubleshooting](#troubleshooting)
7. [Future Work](#future-work)

## Project Overview

The Agentic Underwriting system:
- **Analyzes Loan Applications:** Users provide loan application details through a Streamlit interface.
- **Automates Compliance Checks:** Uses FinBERT for classification and a RAG pipeline for generating underwriting guidelines.
- **Leverages State-of-the-Art Models:** Implements GPT-J-6B for text generation, with options for GPU acceleration and 8-bit quantization if a CUDA-enabled GPU is available.


## Prerequisites

- **Python 3.12** (recommended)
- **CUDA-enabled GPU** (optional, but recommended for faster inference; CPU mode is supported)
- **Git**

## Step-by-Step Setup

### 1. Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/your-username/agentic-underwriting.git
cd agentic-underwriting
```

### 2. Create and Activate a Virtual Environment
It is recommended to isolate dependencies using a virtual environment for Python 3.12.

For Windows:
```bash
"C:\Users\saivi\AppData\Local\Programs\Python\Python312\python.exe" -m venv venv312
venv312\Scripts\activate
```

For macOS/Linux:
```bash
python3.12 -m venv venv312
source venv312/bin/activate
```

### 3. Install Dependencies
a. Install GPU-Compatible PyTorch (Adjust CUDA Version as Needed)
```bash
"C:\Users\saivi\AppData\Local\Programs\Python\Python312\python.exe" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
b. Install Other Required Packages
```bash
"C:\Users\saivi\AppData\Local\Programs\Python\Python312\python.exe" -m pip install transformers accelerate bitsandbytes langchain langchain-community python-dotenv sentence-transformers faiss-cpu
```

## Running the Application
Launch the interactive Streamlit interface with:
```bash
streamlit run app/streamlit_app.py
```
Once the app loads in your browser, enter your loan application details in the text area and click "Analyze Application" to view the results.

## Code Overview
### Models
models/finbert_classifier.py: Contains the FinBERT-based classification logic for financial texts.\
models/rag_pipeline.py: Implements the RAG pipeline:\
Vector Store: Creates a semantic vector store using FAISS and HuggingFace embeddings (e.g., all-mpnet-base-v2).\
Text Generation: Loads GPT-J-6B for generating underwriting guidelines.\
GPU Support: If CUDA is available, the model is loaded with 8-bit quantization (load_in_8bit=True), half-precision (torch.float16), and an offload_folder is specified.\
CPU Mode: If CUDA is not available, the model is loaded in standard precision.\
### Agents
agents/underwriting_agent.py: Orchestrates the workflow by calling functions from the classification and RAG pipeline modules to process the loan application and produce a compliance decision.
### App
app/streamlit_app.py: The entry point for the Streamlit UI.\
Loads environment variables.\
Sets up the vector store.\
Imports the underwriting agent.\
Provides a user interface for entering application details and displaying the analysis results.\
## Troubleshooting
CUDA & BitsandBytes Issues: Ensure your GPU drivers are up-to-date. The code conditionally loads the model using 8-bit quantization if CUDA is available; otherwise, it loads the model in standard precision.\
Offload Folder Error: If you encounter errors regarding offloaded weights, the code creates an offload folder in the project root automatically. Verify that this folder exists or adjust the path if necessary.\
Deprecation Warnings: If you see warnings about deprecated imports from LangChain, update your code to use langchain-community as indicated in the code comments.\
Environment Consistency:  Ensure all commands are run in your Python 3.12 virtual environment (check with python --version).\

## Future Work
Integrate additional models (e.g., GPT-4, Llama 2) if access is granted.\
Enhance multi-agent orchestration with more robust policy checks.\
Extend the UI with additional analytics and visualization features.\


