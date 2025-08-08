import os
import fitz  # PyMuPDF
import cv2
import numpy as np
import torch
import concurrent.futures
import tempfile
import requests
from pdf2image import convert_from_path
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# --- CONFIGURATION ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
MAX_TOKENS_PER_CHUNK = 256
CHUNK_OVERLAP = 30

def download_pdf(pdf_url: str) -> str:
    """
    Download a PDF from the given URL to a temporary file.
    Returns the local file path.
    """
    print(f"Downloading PDF from: {pdf_url}")
    response = requests.get(pdf_url)
    response.raise_for_status()

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_file.write(response.content)
    tmp_file.close()

    print(f"PDF saved to temporary path: {tmp_file.name}")
    return tmp_file.name

def process_page(page_info):
    """
    Takes a tuple (page_number, pdf_path, page_image_pil), performs visual
    extraction, and returns a list of chunk dictionaries for that page.
    """
    page_num, pdf_path, page_image_pil = page_info
    print(f"[Process {os.getpid()}] Starting page {page_num}...")

    page_image = cv2.cvtColor(np.array(page_image_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(binary, kernel, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_num-1)
        img_height, img_width, _ = page_image.shape
        pdf_width, pdf_height = page.rect.width, page.rect.height
        x_scale = pdf_width / img_width
        y_scale = pdf_height / img_height
        words = page.get_text("words")

    page_chunks = []
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    for contour in sorted_contours:
        if cv2.contourArea(contour) < 1000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        blob_bbox = fitz.Rect(x * x_scale, y * y_scale, (x + w) * x_scale, (y + h) * y_scale)
        words_in_blob = [word for word in words if fitz.Rect(word[:4]).intersects(blob_bbox)]
        words_in_blob.sort(key=lambda w: (w[1], w[0]))

        if words_in_blob:
            chunk_text = " ".join([word[4] for word in words_in_blob])
            page_chunks.append({"text": chunk_text, "page_number": page_num})

    print(f"[Process {os.getpid()}] Finished page {page_num}, found {len(page_chunks)} chunks.")
    return page_chunks

def process_and_embed_pdf(pdf_url: str, page_nums_to_process: list[int]):
    """
    Main function to download a PDF from URL, process selected pages,
    embed its content, and upload to Pinecone.

    Args:
        pdf_url (str): The URL to the PDF document.
        page_nums_to_process (list[int]): A list of 1-indexed page numbers to process.
    """
    
    if not all([PINECONE_API_KEY, PINECONE_HOST, PINECONE_INDEX_NAME]):
        print("⚠️ Pinecone environment variables not set. Exiting.")
        return

    # --- STEP 0: Download PDF ---
    pdf_path = download_pdf(pdf_url)

    print(f"printing the number of pages requested {page_nums_to_process}",flush=True)

    print("--- STEP 1: Initializing models and Pinecone ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    embedding_model = SentenceTransformer(MODEL_NAME)

    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if it doesn't exist
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=embedding_model.get_sentence_embedding_dimension(),
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

    index = pc.Index(PINECONE_INDEX_NAME)
    print("Pinecone setup complete.")

    print("\n--- STEP 2: Converting PDF pages to images ---")
    pdf_images_pil = convert_from_path(
        pdf_path,
        first_page=min(page_nums_to_process),
        last_page=max(page_nums_to_process)
    )
    page_image_map = {i: img for i, img in zip(page_nums_to_process, pdf_images_pil)}

    print("\n--- STEP 3: Processing pages in parallel to extract chunks ---")
    pages_to_process_info = [(num, pdf_path, page_image_map[num]) for num in page_nums_to_process]
    all_pages_chunks = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_page, pages_to_process_info)
        for page_result in results:
            all_pages_chunks.extend(page_result)

    print(f"\n✅ Total chunks extracted from all pages: {len(all_pages_chunks)}")

    print("\n--- STEP 4: Splitting oversized chunks ---")
    final_chunks_with_metadata = []
    for chunk_info in all_pages_chunks:
        tokens = tokenizer.encode(chunk_info['text'], add_special_tokens=False)
        if len(tokens) <= MAX_TOKENS_PER_CHUNK:
            final_chunks_with_metadata.append(chunk_info)
            continue

        step = MAX_TOKENS_PER_CHUNK - CHUNK_OVERLAP
        for i in range(0, len(tokens), step):
            sub_chunk_tokens = tokens[i: i + MAX_TOKENS_PER_CHUNK]
            sub_chunk_text = tokenizer.decode(sub_chunk_tokens)
            final_chunks_with_metadata.append({"text": sub_chunk_text, "page_number": chunk_info['page_number']})

    print(f"Total chunks after splitting: {len(final_chunks_with_metadata)}")

    print("\n--- STEP 5: Generating embeddings and uploading to Pinecone ---")
    texts_to_embed = [chunk['text'] for chunk in final_chunks_with_metadata]
    embeddings = embedding_model.encode(texts_to_embed, show_progress_bar=True)

    vectors_to_upsert = []
    for i, (chunk, embedding) in enumerate(zip(final_chunks_with_metadata, embeddings)):
        vector_id = f"vec_{os.path.basename(pdf_path)}_{i}"
        metadata = {
            "text": chunk['text'],
            "page_number": chunk['page_number'],
            "original_document": pdf_url
        }
        vectors_to_upsert.append((vector_id, embedding.tolist(), metadata))

    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i: i + batch_size]
        index.upsert(vectors=batch)
        print(f"Uploaded batch {i // batch_size + 1} to Pinecone.")

    print("\n✅ Embedding and upload process complete!")
