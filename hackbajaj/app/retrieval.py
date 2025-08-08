# retrieval.py

import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pinecone import Pinecone

# --- CONFIGURATION ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

def retrieval_answer(queries: list[str], top_k: int = 3) -> list[str]:
    """
    Takes a list of queries, retrieves relevant context from Pinecone,
    and generates answers using the Gemini API.

    Args:
        queries (list[str]): A list of questions to answer.
        top_k (int): The number of relevant chunks to retrieve for context.

    Returns:
        list[str]: A list of answers corresponding to the input queries.
    """
    if not all([GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME]):
        print("⚠️ API keys or Pinecone index name not set. Exiting.")
        return []

    print("--- Initializing models and connecting to services... ---")
    try:
        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        llm_model = genai.GenerativeModel('gemini-2.0-flash-lite')

        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
             raise ValueError(f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist.")
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Initialize the same embedding model used in embedder.py
        embedding_model = SentenceTransformer(MODEL_NAME)
        print("✅ Initialization complete.")

    except Exception as e:
        print(f"An error occurred during initialization: {e}")
        return []

    final_answers = []
    count = 0
    for query in queries:
        print(f"\n--- Processing Query: '{query}' ---")
        
        # 1. Embed the query
        query_embedding = embedding_model.encode(query).tolist()
        
        # 2. Retrieve from Pinecone
        print(f"Searching for top {top_k} relevant chunks...")
        response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # 3. Generate the Answer with LLM
        context_chunks = [match['metadata']['text'] for match in response['matches']]
        context_for_llm = "\n---\n".join(context_chunks)
        
        prompt = (f"Based ONLY on the context provided below, please answer the question. "
                  f"Do not use any prior knowledge. If the context does not contain the answer, "
                  f"say 'The answer is not available in the provided context.'\n\n"
                  f"Context:\n{context_for_llm}\n\n"
                  f"Question:\n{query}")
        
        try:
            print("Sending prompt to Gemini API...")
            llm_response = llm_model.generate_content(prompt)
            final_answers.append(llm_response.text.strip())
            print(f"💡 Answer: {llm_response.text.strip()}")
            count += 1
            if count == len(queries):
                print(f"All queries answered - Printing {final_answers}") 
        except Exception as e:
            error_message = f"An error occurred with the Gemini API: {e}"
            print(error_message)
            final_answers.append(error_message)
            
    return final_answers


'''
if __name__ == '__main__':
    # Define a dictionary or list of questions to ask
    questions = [
        "What is the grace period for premium payments?",
        "What does 'break in policy' mean?",
        "Can the policyholder cancel his/her policy at any time?"
    ]
    
    # Get the answers
    answers = retrieval_answer(queries=questions, top_k=3)
    
    print("\n\n--- FINAL RESULTS ---")
    for i, (q, a) in enumerate(zip(questions, answers)):
        print(f"{i+1}. Question: {q}")
        print(f"   Answer: {a}\n")
'''