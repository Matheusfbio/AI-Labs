# 📖 Visão Geral

AI Labs é uma interface de chat com IA rodando no Hugging Face Spaces, usando o modelo Qwen2.5-0.5B-Instruct com suporte a RAG (Retrieval-Augmented Generation).

## O que o projeto faz

- Permite conversar com um modelo de linguagem via interface web
- Aceita upload de documentos `.txt` e `.pdf` para enriquecer as respostas com contexto
- Usa ZeroGPU para aceleração gratuita de GPU no Hugging Face Spaces

## Estrutura de arquivos

```
AI-Labs/
├── app.py              # Aplicação principal
├── requirements.txt    # Dependências Python
├── Dockerfile          # Container para rodar localmente
├── README.md           # Configuração do Hugging Face Space
└── docs/
    ├── 01-overview.md       # Este arquivo
    ├── 02-architecture.md   # Arquitetura e fluxo
    ├── 03-rag.md            # Como o RAG funciona
    ├── 04-model.md          # Modelo e parâmetros de geração
    └── 05-deploy.md         # Como rodar e fazer deploy
```

## Links

- [Hugging Face Space](https://huggingface.co/spaces/Matheus-fbio/AI-Labs)
- [Modelo Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Documentação ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)
