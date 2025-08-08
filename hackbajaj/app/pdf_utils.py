# --- pdf_utils.py ---

from typing import List, Tuple
import requests
import fitz  # PyMuPDF
import tempfile
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def download_pdf(pdf_url: str) -> str:
    """Download the PDF from the given URL and return the local file path."""
    response = requests.get(pdf_url)
    response.raise_for_status()
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_file.write(response.content)
    tmp_file.close()
    return tmp_file.name


def extract_text_by_page(pdf_path: str) -> List[str]:
    """Extract text from each page of the PDF."""
    doc = fitz.open(pdf_path)
    return [page.get_text() for page in doc]


def score_pages_by_questions(pages: List[str], questions: List[str], top_k: int = 3) -> List[int]:
    """
    For each question, find top_k most relevant page numbers (1-indexed) using TF-IDF.
    Return the union of all such pages as a sorted list without duplicates.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(pages)


    all_top_pages = set()


    for question in questions:
        question_vector = vectorizer.transform([question])
        scores = (tfidf_matrix @ question_vector.T).toarray().flatten()
        top_indices = np.argsort(scores)[-top_k:][::-1]  # Top-k indices in descending order
        all_top_pages.update(idx + 1 for idx in top_indices)  # Convert to 1-based indexing
    return sorted(all_top_pages)


def download_pdf_and_match_pages(pdf_url: str, expanded_questions: List[str]) -> Tuple[List[int], str]:
    """
    Download the PDF, extract text, and return union of top 3 pages per question (1-indexed),
    along with the local file path.
    """
    local_pdf_path = download_pdf(pdf_url)
    pages = extract_text_by_page(local_pdf_path)
    all_top_pages = score_pages_by_questions(pages, expanded_questions, top_k=3)
    pages=[int(p) for p in all_top_pages]
    top_pages=sorted(pages)
    print(top_pages)
    return top_pages, local_pdf_path
