import streamlit as st
from agents.underwriting_agent import underwriting_agent
from models.rag_pipeline import setup_vector_store

# For demonstration, assume you have a list of underwriting guidelines
documents = [
    "Guideline 1: Loan amounts above $50K require additional verification.",
    "Guideline 2: Applicants with low credit scores must provide collateral.",
    # ... add more guidelines as needed
]

# Set up the vector store (could be done outside the main loop for efficiency)
vector_store = setup_vector_store(documents)

def main():
    st.title("Agentic Underwriting: Loan & Compliance Workflow")
    application_text = st.text_area("Enter Loan Application Details:")
    
    if st.button("Analyze Application"):
        with st.spinner("Processing..."):
            result = underwriting_agent(application_text, vector_store)
        st.subheader("Analysis Result")
        st.write("Classification Scores:", result["classification"])
        st.write("Relevant Guidelines:", result["guidelines"])
        st.write("Compliance Decision:", result["compliance"])

if __name__ == "__main__":
    main()
