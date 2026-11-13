# 🚀 Deploy e Execução

## Rodar localmente

### Com Python direto

```bash
pip install -r requirements.txt
python app.py
```

Acesse em `http://localhost:7860`

> O ZeroGPU não funciona localmente. O decorator `@spaces.GPU` é ignorado fora do Hugging Face Spaces — o modelo roda em CPU ou GPU local normalmente.

### Com Docker

```bash
docker build -t ai-labs .
docker run -p 7860:7860 ai-labs
```

## Deploy no Hugging Face Spaces

### Pré-requisitos

- Conta no [Hugging Face](https://huggingface.co)
- Token de acesso com permissão `write` em [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Configurar o remote

```bash
git remote set-url origin https://<usuario>:<token>@huggingface.co/spaces/<usuario>/AI-Labs
```

### Enviar alterações

```bash
git add .
git commit -m "sua mensagem"
git push origin main
```

O Space reinicia automaticamente após o push.

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | Modelo de geração de texto |

Configure em **Settings → Variables and secrets** no painel do Space.

## Dependências

| Pacote | Uso |
|---|---|
| `gradio` | Interface web |
| `spaces` | Integração com ZeroGPU |
| `torch` | Backend de inferência |
| `transformers` | Carregamento do modelo Qwen |
| `accelerate` | Otimização de carregamento com `device_map` |
| `faiss-cpu` | Índice vetorial para RAG |
| `sentence-transformers` | Embeddings para RAG |
| `pypdf` | Extração de texto de PDFs |
