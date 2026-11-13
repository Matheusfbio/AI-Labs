!pip -q install -U \
    transformers \
    accelerate \
    sentence-transformers \
    datasets \
    evaluate \
    huggingface_hub \
    langchain \
    langchain-core \
    langchain-huggingface

     


import torch
import platform

print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "Memória GPU:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "GB"
    )
else:
    print("Executand
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto"
)

print("Modelo carregado!")
print("Dispositivo:", model.device)
o em CPU")

     
Python: 3.13.16
PyTorch: 2.14.1+cu130
CUDA disponível: True
GPU: Tesla T4
Memória GPU: 14.56 GB


import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto"
)

print("Modelo carregado!")
print("Dispositivo:", model.device)

     

import torch

# Função que envia a mensagem para o modelo e retorna a resposta gerada
def chat(message, max_tokens=300):
    # Define o histórico/formato da conversa seguindo os papéis (roles) suportados pelo modelo
    messages = [
        {
            "role": "system",
            "content": "Você é um assistente especializado em tecnologia."
        },
        {
            "role": "user",
            "content": message
        }
    ]

    # Aplica o template de chat específico do modelo (adicionando os tokens especiais de início/fim de turno)
    # E converte tudo para tensores do PyTorch (.to(model.device) move os dados para a GPU/CPU onde o modelo está)
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    # Desabilita o cálculo de gradientes para economizar memória e acelerar a inferência
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,        # Permite amostragem probabilística para respostas mais criativas/variadas
            temperature=0.7,       # Controla a criatividade (valores menores são mais deterministas)
            top_p=0.9,             # Filtra as palavras mais prováveis que somam 90% de probabilidade
            pad_token_id=tokenizer.eos_token_id
        )

    # Remove os tokens de entrada da resposta gerada para obter apenas o que o modelo respondeu
    generated_tokens = outputs[
        :, inputs["input_ids"].shape[1]:
    ]

    # Decodifica os tokens numéricos gerados de volta para texto legível em português
    response = tokenizer.decode(
        generated_tokens[0],
        skip_special_tokens=True
    )

    return response.strip()

# Loop contínuo para manter a conversa ativa
print("Iniciando o Chat contínuo. Digite 'sair', 'exit' ou 'parar' para encerrar.\n")

while True:
    pergunta_usuario = input("Você: ")

    # Verifica se o usuário deseja encerrar o loop
    if pergunta_usuario.strip().lower() in ["sair", "exit", "parar"]:
        print("\nConversa encerrada. Até a próxima!")
        break

    if not pergunta_usuario.strip():
        continue

    print("\nProcessando sua resposta...\n")
    response = chat(pergunta_usuario)

    print(f"Modelo: {response}")
    print("-" * 50 + "\n")