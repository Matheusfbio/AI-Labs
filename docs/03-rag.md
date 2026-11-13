# 🔍 RAG — Retrieval-Augmented Generation

RAG é uma técnica que permite ao modelo responder com base em documentos fornecidos pelo usuário, sem precisar de fine-tuning.

## Como funciona

### 1. Indexação (ao enviar arquivos)

```
Arquivo .txt ou .pdf
        │
        ▼
  Extração do texto
        │
        ▼
  chunk_text() — divide em blocos de 500 caracteres
        │
        ▼
  embedder.encode() — converte cada chunk em vetor float32
        │
        ▼
  faiss.IndexFlatL2.add() — armazena os vetores no índice
```

### 2. Recuperação (a cada mensagem)

```
Mensagem do usuário
        │
        ▼
  embedder.encode(query) — gera vetor da pergunta
        │
        ▼
  rag_index.search(query_vec, k=3) — busca os 3 chunks mais similares
        │
        ▼
  Trechos retornados são injetados no system prompt
```

## Por que FAISS?

- Busca por similaridade vetorial muito rápida
- Funciona 100% local, sem API externa
- `IndexFlatL2` usa distância euclidiana, simples e eficaz para textos curtos

## Por que all-MiniLM-L6-v2?

- Modelo leve (22M parâmetros)
- Gera embeddings de 384 dimensões
- Ótimo custo-benefício entre qualidade e velocidade

## Limitações

- O índice é mantido em memória — reiniciar o Space apaga os documentos
- Chunks de 500 caracteres podem cortar frases no meio
- Sem reranking — os 3 primeiros resultados por distância L2 são usados diretamente
