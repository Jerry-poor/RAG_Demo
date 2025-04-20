# build_faiss_index.py
import os
import pickle
import numpy as np
import torch
import faiss
import chardet
from transformers import AutoTokenizer, AutoModel

# Configuration
MODEL_ID    = "ibm-granite/granite-embedding-107m-multilingual"
SAVE_DIR    = "../granite_model"        
KNOW_DIR    = "knowledge"               
INDEX_FILE  = "knowledge_chunks.index"     
META_FILE   = "knowledge_chunks_meta.pkl" 
MAX_TOKENS  = 512
STRIDE      = 128
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Load embedding model
tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
model     = AutoModel.from_pretrained(SAVE_DIR).to(DEVICE)
model.eval()


def chunk_text(text, max_length=MAX_TOKENS, stride=STRIDE):
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


texts    = []
metadata = []  # 存 (path, chunk_idx, chunk_text)
for root, _, files in os.walk(KNOW_DIR):
    for fname in files:
        path = os.path.join(root, fname)
        with open(path, "rb") as f:
            raw = f.read()
        enc = chardet.detect(raw)["encoding"] or "utf-8"
        content = raw.decode(enc, errors="replace")
        # 切分
        for idx, chunk in enumerate(chunk_text(content)):
            texts.append(chunk)
            metadata.append((path, idx, chunk))

#embeddings
embeddings = []
with torch.no_grad():
    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_TOKENS
        ).to(DEVICE)
        out   = model(**enc).last_hidden_state       # [1, L, D]
        mask  = enc["attention_mask"].unsqueeze(-1)  # [1, L, 1]
        summed = (out * mask).sum(1)                  # [1, D]
        counts = mask.sum(1)                          # [1, 1]
        emb    = (summed / counts).squeeze(0).cpu().numpy()
        embeddings.append(emb.astype(np.float32))

emb_array = np.vstack(embeddings)

#Faiss
index = faiss.IndexFlatL2(emb_array.shape[1])
index.add(emb_array)
faiss.write_index(index, INDEX_FILE)

with open(META_FILE, "wb") as f:
    pickle.dump(metadata, f)

print(f"Built index with {len(texts)} chunks.")
print("Index saved to", INDEX_FILE)
print("Metadata saved to", META_FILE)
