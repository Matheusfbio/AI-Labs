# 🏗️ Arquitetura

## Fluxo de uma mensagem

```
Usuário digita mensagem
        │
        ▼
  retrieve(query)
  ┌─────────────────────────────────┐
  │ 1. Gera embedding da query      │
  │ 2. Busca os 3 chunks mais       │
  │    próximos no índice FAISS     │
  │ 3. Retorna os trechos           │
  └─────────────────────────────────┘
        │
        ▼
  Monta o prompt com contexto
  ┌─────────────────────────────────┐
  │ system prompt + contexto RAG   │
  │ + histórico da conversa        │
  │ + mensagem do usuário          │
  └─────────────────────────────────┘
        │
        ▼
  generate_reply() [@spaces.GPU]
  ┌─────────────────────────────────┐
  │ Tokeniza com apply_chat_template│
  │ Roda model.generate()          │
  │ Decodifica os tokens de saída  │
  └─────────────────────────────────┘
        │
        ▼
  Resposta exibida no chat
```

## Componentes principais

| Componente | Responsabilidade |
|---|---|
| `embedder` | Converte texto em vetores numéricos |
| `rag_index` | Índice FAISS para busca por similaridade |
| `rag_chunks` | Lista de trechos dos documentos indexados |
| `model` | Gera a resposta em linguagem natural |
| `tokenizer` | Converte texto ↔ tokens para o modelo |

## Modelos usados

| Papel | Modelo |
|---|---|
| Geração de texto | `Qwen/Qwen2.5-0.5B-Instruct` |
| Embeddings (RAG) | `sentence-transformers/all-MiniLM-L6-v2` |
