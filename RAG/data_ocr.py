#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_ocr.py

- Traverses the `knowledge/` directory.
- For each file:
  - If it's an image (jpg/png/etc), runs OCR to extract text.
  - Otherwise, auto-detects encoding and reads its text content.
- Chunks every extracted text into overlapping pieces.
- Embeds each chunk with the ibm-granite model.
- Builds a FAISS index over all chunk embeddings.
- Saves both the index and metadata (path, chunk_idx, chunk_text) to disk.
"""

import os
import pickle
import numpy as np
import torch
import faiss
import chardet
from PIL import Image
import easyocr
from transformers import AutoTokenizer, AutoModel

# Configuration
MODEL_DIR   = "../granite_model"
KNOW_DIR    = "knowledge"
INDEX_FILE  = "knowledge_ocr_chunks.index"
META_FILE   = "knowledge_ocr_chunks_meta.pkl"
MAX_TOKENS  = 512
STRIDE      = 128
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
OCR_LANGS   = ['en','ch_sim']  # adjust to your document languages

# Initialize OCR reader
ocr_reader = easyocr.Reader(OCR_LANGS, gpu=torch.cuda.is_available())

# Load embedding model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModel.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

def chunk_text(text, max_length=MAX_TOKENS, stride=STRIDE):
    """Split text into overlapping chunks of up to max_length tokens."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start  = 0
    while start < len(tokens):
        end = min(start + max_length, len(tokens))
        chunk_ids = tokens[start:end]
        chunks.append(
            tokenizer.decode(chunk_ids, clean_up_tokenization_spaces=True)
        )
        if end == len(tokens):
            break
        start += max_length - stride
    return chunks

def extract_text(path):
    """Run OCR if image, else read text with charset detection."""
    ext = os.path.splitext(path)[1].lower()
    if ext in {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}:
        # OCR path
        img = Image.open(path)
        # easyocr returns list of (bbox, text, conf)
        result = ocr_reader.readtext(np.array(img), detail=0)
        return "\n".join(result)
    else:
        # Text file path
        with open(path, 'rb') as f:
            raw = f.read()
        enc = chardet.detect(raw)['encoding'] or 'utf-8'
        return raw.decode(enc, errors='replace')

def build_index():
    texts   = []
    metadata= []  # list of (path, chunk_idx, chunk_text)

    # 1. Traverse and extract
    for root, _, files in os.walk(KNOW_DIR):
        for fname in files:
            path = os.path.join(root, fname)
            print(f"Processing {path} ...")
            try:
                full_text = extract_text(path)
            except Exception as e:
                print(f"   Failed to extract {path}: {e}")
                continue

            # 2. Chunk
            for idx, chunk in enumerate(chunk_text(full_text)):
                texts.append(chunk)
                metadata.append((path, idx, chunk))

    # 3. Embed
    embeddings = []
    with torch.no_grad():
        for chunk in texts:
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=MAX_TOKENS
            ).to(DEVICE)
            out    = model(**enc).last_hidden_state
            mask   = enc["attention_mask"].unsqueeze(-1)
            summed = (out * mask).sum(1)
            counts = mask.sum(1)
            emb    = (summed / counts).squeeze(0).cpu().numpy().astype(np.float32)
            embeddings.append(emb)

    emb_array = np.vstack(embeddings)

    # 4. Build & save FAISS index
    index = faiss.IndexFlatL2(emb_array.shape[1])
    index.add(emb_array)
    faiss.write_index(index, INDEX_FILE)

    # 5. Save metadata
    with open(META_FILE, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\nBuilt index with {len(texts)} chunks.")
    print("Index saved to", INDEX_FILE)
    print("Metadata saved to", META_FILE)

if __name__ == "__main__":
    build_index()
