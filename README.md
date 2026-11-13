---
title: AI Labs
emoji: 🤖
colorFrom: purple
colorTo: red
sdk: gradio
sdk_version: 5.29.0
python_version: '3.12'
app_file: app.py
pinned: false
---

# 🤖 AI Labs — Chat com Qwen 2.5

Interface de chat com o modelo [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) rodando com ZeroGPU no Hugging Face Spaces.

## Funcionalidades

- Chat com histórico de conversa
- Configuração de parâmetros de geração (tokens, temperatura, top-p)
- Aceleração via ZeroGPU (sem custo de GPU dedicada)
- Prompt de sistema focado em tecnologia

## Como rodar localmente

```bash
pip install -r requirements.txt
python app.py
```

Acesse em `http://localhost:7860`

## Parâmetros

| Parâmetro | Padrão | Descrição |
|---|---|---|
| Máximo de tokens | 300 | Tamanho máximo da resposta |
| Temperatura | 0.7 | Criatividade da resposta (0 = determinístico) |
| Top-p | 0.9 | Diversidade do vocabulário amostrado |

## Modelo

O modelo padrão é `Qwen/Qwen2.5-0.5B-Instruct`, configurável via variável de ambiente `MODEL_ID`.

## Stack

- [Gradio](https://gradio.app) — interface
- [Transformers](https://huggingface.co/docs/transformers) — inferência
- [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) — aceleração gratuita
