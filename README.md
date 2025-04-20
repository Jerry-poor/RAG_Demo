# RAG_Demo 🧠🔍  
一个小小的娱乐项目 🎮✨  

This is a lightweight demo project for exploring Retrieval-Augmented Generation (RAG).  
这是我用来探索检索增强生成（RAG）的轻量化演示项目。  

I use FAISS to build a simple vector database. 📦  
我使用 FAISS 搭建了一个简易的向量数据库。  

The embedding model is `ibm-granite/granite-embedding-107m-multilingual`, which supports multiple languages. 🌍  
使用的 embedding 模型是 `ibm-granite/granite-embedding-107m-multilingual`，支持多种语言。  

The LLM backend is powered by DeepSeek-R1. 🚀  
大语言模型部分使用的是 DeepSeek-R1。  

Note: Neither DeepSeek-R1 nor the IBM embedding model supports multi-modal input. 🙅‍♂️🖼️  
注意：DeepSeek-R1 和 IBM 的 embedding 模型都不支持多模态输入。  

So the multi-modal feature is currently just a placeholder for future extension. 🧩  
因此目前的多模态部分只是一个框架，未来有需要时会继续扩展。  

I integrated a lightweight OCR tool to simulate basic multi-modal functionality. 📄➡️🔤  
我使用了一个轻量级的 OCR 库来做一个简单的多模态演示。  

Running large models locally can be computationally expensive. 🧮💻  
本地运行大模型的算力开销较大。  

For now, I use DeepSeek’s API to build this demo and practice workflow. 🔧📡  
目前我通过调用 DeepSeek API 来做这个演示项目，也顺便练习一下流程。  

I might consider full local deployment in the future if needed. 🏠🤔  
将来有需要的话，我会考虑做全本地部署。  
