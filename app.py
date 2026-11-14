from __future__ import annotations

import os
import json
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()

SYSTEM_PROMPT = """Você é um extrator de entidades. Dado um texto, extraia as entidades e retorne APENAS um JSON válido no formato:
{
  "pessoas": [],
  "organizações": [],
  "lugares": [],
  "datas": [],
  "tecnologias": []
}
Não adicione explicações, apenas o JSON."""


def extract_entities(text: str) -> tuple[str, str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Texto:\n{text}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    answer_tokens = output[0, inputs["input_ids"].shape[-1]:]
    raw = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

    try:
        # extrai o bloco JSON mesmo que venha com texto ao redor
        start, end = raw.index("{"), raw.rindex("}") + 1
        data = json.loads(raw[start:end])
        formatted = "\n".join(
            f"**{k.capitalize()}:** {', '.join(v) if v else '—'}"
            for k, v in data.items()
        )
    except (ValueError, json.JSONDecodeError):
        formatted = raw

    return raw, formatted


EXAMPLES = [
    "Elon Musk fundou a SpaceX em 2002 em Hawthorne, Califórnia. A empresa usa Python e C++ em seus sistemas de telemetria.",
    "A Amazon lançou o AWS re:Invent em Las Vegas em dezembro de 2023, apresentando novidades no SageMaker e no Bedrock.",
    "Marie Curie nasceu em Varsóvia em 1867 e ganhou o Prêmio Nobel em Paris pela descoberta do rádio e polônio.",
]

with gr.Blocks(title="Extrator de Entidades — Qwen 2.5") as demo:
    gr.Markdown("# 🔍 Extrator de Entidades — Qwen 2.5")
    gr.Markdown("Cole um texto e extraia automaticamente pessoas, organizações, lugares, datas e tecnologias.")

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(lines=6, placeholder="Cole seu texto aqui…", label="Texto de entrada")
            gr.Examples(examples=EXAMPLES, inputs=text_input)
            extract_btn = gr.Button("Extrair entidades", variant="primary")
        with gr.Column(scale=2):
            result_md = gr.Markdown(label="Entidades extraídas")
            with gr.Accordion("JSON bruto", open=False):
                result_raw = gr.Textbox(label="Resposta do modelo", interactive=False)

    extract_btn.click(extract_entities, inputs=text_input, outputs=[result_raw, result_md])

demo.launch()
