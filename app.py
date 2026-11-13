"""Aplicativo Streamlit para conversar com o modelo Qwen do notebook.

Execute com: streamlit run app.py
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
SYSTEM_PROMPT = "Você é um assistente especializado em tecnologia."
STOP_WORDS = {"sair", "parar", "exit", "quit"}


@st.cache_resource(show_spinner="Carregando o modelo e o tokenizador…")
def load_model(model_id: str) -> tuple[Any, Any]:
    """Carrega o modelo uma vez e o mantém em cache entre interações."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def generate_reply(
    prompt: str,
    history: list[dict[str, str]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Gera uma resposta com o template de conversa do tokenizador Qwen."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    # device_map='auto' pode distribuir o modelo em mais de um dispositivo;
    # os tensores de entrada devem ir para o dispositivo de entrada do modelo.
    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    generation_args: dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_args.update(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        generation_args["do_sample"] = False

    with torch.inference_mode():
        output = model.generate(**generation_args)

    answer_tokens = output[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()


st.set_page_config(page_title="Laboratório de IA — Qwen", page_icon="🤖")
st.title("🤖 Laboratório de IA — Qwen 2.5")
st.caption("Chat local com Qwen2.5-0.5B-Instruct, baseado no notebook do repositório.")

with st.sidebar:
    st.header("Configurações")
    model_id = st.text_input("Modelo Hugging Face", value=MODEL_ID)
    max_new_tokens = st.slider("Máximo de tokens na resposta", 32, 1024, 300, step=32)
    temperature = st.slider("Temperatura", 0.0, 1.5, 0.7, step=0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)
    st.caption(
        "O modelo é baixado do Hugging Face na primeira execução. "
        "Uma GPU compatível acelera a geração; em CPU, a resposta pode demorar."
    )
    if torch.cuda.is_available():
        st.success(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.info("CUDA não detectada — o modelo será executado em CPU.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Digite sua mensagem… (ou 'sair' para limpar a conversa)")
if prompt:
    if prompt.strip().lower() in STOP_WORDS:
        st.session_state.messages = []
        st.rerun()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        tokenizer, model = load_model(model_id.strip())
        # Envia ao modelo o histórico anterior, sem duplicar a mensagem atual.
        previous_history = st.session_state.messages[:-1]
        with st.chat_message("assistant"):
            with st.spinner("Gerando resposta…"):
                answer = generate_reply(
                    prompt,
                    previous_history,
                    tokenizer,
                    model,
                    max_new_tokens,
                    temperature,
                    top_p,
                )
            st.markdown(answer or "Não foi possível gerar uma resposta.")
        st.session_state.messages.append(
            {"role": "assistant", "content": answer or "Não foi possível gerar uma resposta."}
        )
    except Exception as exc:
        # Remove a mensagem para não deixar o histórico inconsistente; não expõe
        # detalhes de credenciais ou do ambiente na interface.
        st.session_state.messages.pop()
        st.error(f"Não foi possível carregar o modelo ou gerar a resposta: {exc}")

if st.session_state.messages and st.sidebar.button("Limpar conversa"):
    st.session_state.messages = []
    st.rerun()