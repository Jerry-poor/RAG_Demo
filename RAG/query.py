# -*- coding: utf-8 -*-
"""
Interactive FAISS + DeepSeek Chat Query
- Embeds only the user query with a local ibm-granite model.
- Retrieves top-k precomputed chunks from a FAISS index.
- Constructs a chat-style message list and calls DeepSeek via the OpenAI‑compatible SDK.
- Prints the assistant’s reply to the console.
"""
import os
# work around OpenMP runtime conflicts on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import pickle
import numpy as np
import torch
import faiss
import chardet
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI

# --- Configuration ---
EMB_MODEL_DIR   = "../granite_model"            
INDEX_FILE      = "knowledge_chunks.index"      
EMB_META_PKL    = "knowledge_chunks_meta.pkl"    
API_JSON        = "api.json"                  
TOP_K           = 5
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# --- Local prompt template ---
system_template = (
    "你是一个专业的知识查询助手。\n"
    "以下是从知识库中检索到的文档片段，请结合这些片段回答下面的问题："
)

# --- Load API config ---
with open(API_JSON, 'r', encoding='utf-8') as f:
    api_cfg = json.load(f)
api = api_cfg.get("apis", [])[0] if api_cfg.get("apis") else {}
base_url   = api.get("base_url", "").rstrip('/')
api_key    = api.get("api_key", "")
model_name = api.get("model", "")

# initialize OpenAI‑compatible client for DeepSeek
client = OpenAI(api_key=api_key, base_url=base_url)

# --- Load FAISS index ---
index = faiss.read_index(INDEX_FILE)

# --- Load metadata (expects list of triples) ---
with open(EMB_META_PKL, 'rb') as f:
    metadata = pickle.load(f)
# metadata: List[ (path, chunk_idx, chunk_text) ]

# --- Load embedding model (only for query) ---
tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL_DIR)
model     = AutoModel.from_pretrained(EMB_MODEL_DIR).to(DEVICE)
model.eval()

# --- Embed only the user query ---
def embed_query(query: str) -> np.ndarray:
    enc = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(DEVICE)
    with torch.no_grad():
        out = model(**enc).last_hidden_state
    mask   = enc["attention_mask"].unsqueeze(-1)
    summed = (out * mask).sum(1)
    counts = mask.sum(1)
    emb    = (summed / counts).squeeze(0).cpu().numpy().astype(np.float32)
    return emb / np.linalg.norm(emb)

# --- Retrieve top-k chunks via FAISS ---
def search_topk(query_emb: np.ndarray, k=TOP_K):
    D, I = index.search(query_emb.reshape(1, -1), k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        path, chunk_idx, chunk_text = metadata[idx]
        # 对 chunk_text 做安全截断，避免 prompt 过长
        if len(chunk_text) > 1000:
            chunk_text = chunk_text[:1000] + "…"
        results.append({
            "path": path,
            "chunk_idx": chunk_idx,
            "text": chunk_text,
            "score": float(dist)
        })
    return results

# --- Interactive Loop ---
print("--- Interactive FAISS + DeepSeek Chat Query ---\nType 'exit' to quit.")
while True:
    q = input("Enter your query: ").strip()
    if q.lower() in ("exit", "quit"):
        break

    # 1. embed query & retrieve
    q_emb       = embed_query(q)
    topk_chunks = search_topk(q_emb)

    # 2. build chat messages
    messages = []
    # system role with retrieved context
    context = "\n\n".join(c["text"] for c in topk_chunks)
    messages.append({"role": "system", "content": f"{system_template}\n\n{context}"})
    # user role with original query
    messages.append({"role": "user",   "content": q})

    # 3. call DeepSeek via OpenAI‑compatible SDK
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False
        )
        reply = response.choices[0].message.content
    except Exception as e:
        print(" API 请求失败:", e)
        reply = ""

    # 4. output
    print("\n--- Response ---")
    print(reply)
    print()

print("Goodbye!")
