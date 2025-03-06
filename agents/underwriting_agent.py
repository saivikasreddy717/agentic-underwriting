from models.finbert_classifier import classify_document
from models.rag_pipeline import get_guidelines

def underwriting_agent(application_text: str, vector_store):
    # Step 1: Classify the document using FinBERT
    classification = classify_document(application_text)
    
    # Step 2: Retrieve relevant underwriting guidelines
    query = f"Retrieve guidelines for a loan application: {application_text[:100]}"
    guidelines = get_guidelines(query, vector_store)
    
    # Step 3: Combine the outputs to generate a final compliance decision
    # (Here you can add additional logic or chain more agents)
    decision = {
        "classification": classification,
        "guidelines": guidelines,
        "compliance": "Pass" if classification[0][0] > 0.7 else "Review"  # Example condition
    }
    return decision
