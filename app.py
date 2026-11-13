from __future__ import annotations

import os
import spaces
import torch
import gradio as gr
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SYSTEM_PROMPT = "Você é um assistente especializado em tecnologia."
CHUNK_SIZE = 500

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
model.eval()
embedder = SentenceTransformer(EMBED_MODEL_ID)

# Estado global do índice RAG
rag_index: faiss.IndexFlatL2 | None = None
rag_chunks: list[str] = []


def chunk_text(text: str) -> list[str]:
    return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE) if text[i:i + CHUNK_SIZE].strip()]


def build_index(files: list) -> str:
    global rag_index, rag_chunks
    rag_chunks = []

    for file in files:
        if file.name.endswith(".pdf"):
            import pypdf
            reader = pypdf.PdfReader(file.name)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            with open(file.name, encoding="utf-8", errors="ignore") as f:
                text = f.read()
        rag_chunks.extend(chunk_text(text))

    if not rag_chunks:
        return "Nenhum conteúdo encontrado nos arquivos."

    embeddings = embedder.encode(rag_chunks, convert_to_numpy=True).astype("float32")
    rag_index = faiss.IndexFlatL2(embeddings.shape[1])
    rag_index.add(embeddings)
    return f"✅ {len(rag_chunks)} trechos indexados de {len(files)} arquivo(s)."


def retrieve(query: str, k: int = 3) -> str:
    if rag_index is None or not rag_chunks:
        return ""
    query_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    _, indices = rag_index.search(query_vec, k)
    return "\n\n".join(rag_chunks[i] for i in indices[0] if i < len(rag_chunks))


@spaces.GPU
def generate_reply(prompt: str, history: list, max_new_tokens: int, temperature: float, top_p: float) -> str:
    context = retrieve(prompt)
    system = SYSTEM_PROMPT
    if context:
        system += f"\n\nUse o contexto abaixo para responder:\n{context}"

    messages = [{"role": "system", "content": system}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": prompt})

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    input_device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    generation_args = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_args.update(temperature=temperature, top_p=top_p)

    with torch.inference_mode():
        output = model.generate(**generation_args)

    answer_tokens = output[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()


with gr.Blocks(title="Laboratório de IA — Qwen") as demo:
    gr.Markdown("# 🤖 Laboratório de IA — Qwen 2.5")
    gr.Markdown("Chat com Qwen2.5-0.5B-Instruct usando ZeroGPU e RAG.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Documentos (RAG)")
            file_upload = gr.File(file_types=[".txt", ".pdf"], file_count="multiple", label="Envie arquivos")
            index_btn = gr.Button("Indexar documentos")
            index_status = gr.Textbox(label="Status", interactive=False)
            index_btn.click(build_index, inputs=file_upload, outputs=index_status)

        with gr.Column(scale=3):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(placeholder="Digite sua mensagem…", label="Mensagem")
            with gr.Row():
                max_new_tokens = gr.Slider(32, 1024, value=300, step=32, label="Máximo de tokens")
                temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperatura")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
            clear = gr.Button("Limpar conversa")

    def respond(message, chat_history, max_tokens, temp, tp):
        reply = generate_reply(message, chat_history, max_tokens, temp, tp)
        chat_history.append((message, reply))
        return "", chat_history

    msg.submit(respond, [msg, chatbot, max_new_tokens, temperature, top_p], [msg, chatbot])
    clear.click(lambda: [], outputs=chatbot)

demo.launch()
