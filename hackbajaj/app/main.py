from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import os
import asyncio
import google.generativeai as genai
from app.pdf_utils import download_pdf_and_match_pages
from app.embedder import process_and_embed_pdf
from app.retrieval import retrieval_answer
from dotenv import load_dotenv
load_dotenv()
# --- Initialize FastAPI ---
app = FastAPI()

# --- Configure Gemini ---
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# --- Request Model ---
class QARequest(BaseModel):
    documents: str
    questions: List[str]

# --- Response Model ---
class QAResponse(BaseModel):
    #success: bool
    answers: List[str]
    processing_info: str = "Completed"

# --- Expand Questions with Gemini ---
async def expand_questions(questions: List[str], model_name: str = "models/gemini-2.0-flash-lite") -> List[str]:
    model = genai.GenerativeModel(model_name)

    async def expand(q):
        prompt = f"just generate 5 keywords that bring out the crux of the question. highlight the nouns in the question {q}"
        response = await model.generate_content_async(prompt)
        llm_response = response.text.strip()
        return f"{q} {llm_response}"

    expanded = await asyncio.gather(*(expand(q) for q in questions))
    return expanded

# --- Main Endpoint ---
@app.post("/hackrx/run", response_model=QAResponse)
async def run_hackrx(payload: QARequest, authorization: str = Header(None)):
    # --- Validate Bearer Token ---
    expected_token = os.environ.get("BEARER_TOKEN")
    if not expected_token or authorization != f"Bearer {expected_token}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        # Step 1: Expand questions
        expanded_questions = await expand_questions(payload.questions)

        # Step 2: Get matched pages + local file path from PDF utility
        matched_pages, local_pdf_path = download_pdf_and_match_pages(payload.documents, expanded_questions)

        # Step 3: Pass data to embedder for embedding and storage (no return expected)
        process_and_embed_pdf(payload.documents, matched_pages)

        # Step 4: Generate answers from retrieval module
        answers = retrieval_answer(expanded_questions)

        return QAResponse(
            answers=answers
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
