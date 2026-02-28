# Aegis-RyzenAI-Workspace
Air-gapped enterprise RAG assistant powered by AMD Ryzen AI NPU and iGPU

The Security vs. Productivity Dilemma
Enterprises cannot leverage Generative AI for sensitive corporate documents because uploading data to the cloud violates privacy laws (HIPAA/NDAs), and running AI locally on standard CPUs is too slow.

The Solution
Aegis Workspace is an air-gapped, on-device Retrieval-Augmented Generation (RAG) agent. It securely indexes local documents and runs advanced LLMs completely offline using AMD's heterogeneous compute.

AMD Hardware Orchestration
Vector Embeddings (Background Task): Pinned to the low-power AMD XDNA 2 NPU.

LLM Token Generation (Burst Speed): Handled by the high-bandwidth AMD RDNA 3.5 iGPU via the DirectML Execution Provider.

CPU: Left free for standard enterprise multitasking.

Technology Stack
Hardware: AMD Ryzen AI 400 Series

Software/SDK: AMD Ryzen AI Software 1.7, ONNX Runtime GenAI (OGA) API

AI Models: Quantized Llama-3.2 1B

Framework: Python, LangChain, ChromaDB
