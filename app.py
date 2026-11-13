from __future__ import annotations

import os
import spaces
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
SYSTEM_PROMPT = "Você é um assistente especializado em tecnologia."

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
model.eval()


@spaces.GPU
def generate_reply(prompt: str, history: list, max_new_tokens: int, temperature: float, top_p: float) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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
    gr.Markdown("Chat com Qwen2.5-0.5B-Instruct usando ZeroGPU.")

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
