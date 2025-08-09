# hackrx-bajaj-MVP

**Bajaj Hackathon – Intelligent Policy Document Q&A**  
*Non-commercial use only – Educational, research, and personal tinkering welcome.*

This project was built for the Bajaj Hackathon as an end-to-end intelligent document question-answering system.  
It downloads structured policy PDFs, identifies and ranks relevant sections for given questions, and returns concise answers — all powered by embeddings, retrieval, and generative AI.

-> **Key Features**  
    1. PDF Text Extraction & Ranking – Downloads PDFs, extracts text, and scores pages using TF-IDF relevance against questions expanded using LLM to highlight the crux of the question.  
    2. Custom Chunking – Instead of fixed-size splits, chunks are dynamically sized based on visual structure and text density.  
    3. OpenCV-based Bounding Boxes – We preprocess policy documents by reducing contrast, detecting text regions, and generating bounding boxes. This ensures that spatially related text stays together in chunks. As policy documents are structured and semantic meaning mostly stays spatially clustered.  
    4. MiniLM-L6-v2 Embeddings (384-dim) – High-performance embeddings from a compact transformer model for Pinecone vector search.  
    5. FastAPI Endpoint – A /hackrx/run API for programmatic Q&A.  
    6. Multi-step Retrieval – Combines page scoring and vector search to maximize answer accuracy.  

-> **Requirements**  
    You’ll need the following before running:  
    1. Python 3.10+  
    2. A .env file that includes:  
         PINECONE_API_KEY=your-pinecone-api  
         PINECONE_HOST=https://your-pinecone-host.pinecone.io  
         PINECONE_INDEX_NAME=your-pinecone-index  
         GEMINI_API_KEY=your-gemini-api-key  
         BEARER_TOKEN=your-bearer-token (authenticates your POST request)  
    3. Dependencies:  
         pip install -r requirements.txt  

-> **Running the API**  
    1) Start the FastAPI server:  
       uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload  
       (--host 0.0.0.0 broadcasts it so you can send requests from Postman, curl, or another machine.  
       By default, it’ll run on http://localhost:8000)  
     2) Send a request from your terminal:  
       Example PowerShell command:  
       $response = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/hackrx/run" `  
           -Headers @{Authorization = "Bearer your-bearer-token"} `  
           -Body (@{  
               documents = "https://example.com/policy.pdf"  
               questions = @("What are the eligibility criteria?")  
           } | ConvertTo-Json -Depth 3) `  
           -ContentType "application/json"  
       $response  

-> **How It Works**  
    Question Expansion – Gemini expands each input question into keyword-rich queries for better recall.  
       eg - Processing Query: 'Are the medical expenses for an organ donor covered under this policy?'  
            Here are 5 keywords focusing on the core concepts, with nouns highlighted:  
              1. Medical Expenses  
              2. Organ Donor  
              3. Policy  
              4. Coverage  
              5. Expenses  
    Page Scoring – TF-IDF finds the top-k pages most relevant to the expanded queries.  
    Visual Chunking – OpenCV detects bounding boxes on low-contrast PDF images to preserve spatially close text.  
    Custom Embedding – Each chunk is encoded into a 384-dimensional vector using a MiniLM-L6-v2 model.  
    Vector Search – Pinecone retrieves semantically similar chunks.  
    Answer Generation – Retrieved context is sent to Gemini for concise answers.  

-> **Future Improvements**  
    1. Context Graph Retrieval – Build a graph of related concepts across the document, so chunks from different parts can be linked if relevant to the same question.  
    2. Multi-modal embeddings – Combine visual layout + text embeddings for richer context understanding.  
    3. Re-ranking with LLMs – Use Gemini to re-rank retrieved chunks before answering.  
    4. Support for multiple document formats – DOCX, HTML, etc.  
    5. Faster response times with multiprocessing and reduced latency.  

📜 **License**  
    This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.  
    ✔ Share & modify  
    ✔ Use for research or learning  
    ❌ Not use for commercial purposes  
