# 🧠 Modelo e Parâmetros de Geração

## Modelo

O modelo padrão é `Qwen/Qwen2.5-0.5B-Instruct`, configurável via variável de ambiente:

```bash
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
```

É um modelo instruction-tuned da família Qwen 2.5 da Alibaba, com 0.5B de parâmetros — leve o suficiente para rodar com ZeroGPU.

## System Prompt

```
Você é um assistente especializado em tecnologia.
```

Quando documentos são indexados, o contexto recuperado é anexado automaticamente:

```
Você é um assistente especializado em tecnologia.

Use o contexto abaixo para responder:
<trechos recuperados pelo RAG>
```

## Parâmetros de geração

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `max_new_tokens` | 300 | Limite de tokens gerados na resposta |
| `temperature` | 0.7 | Criatividade — 0 = determinístico, >1 = mais aleatório |
| `top_p` | 0.9 | Amostragem nucleus — considera tokens até 90% da probabilidade acumulada |

### Quando `temperature = 0`

O modelo usa **greedy decoding** (`do_sample=False`), sempre escolhendo o token mais provável. Respostas mais previsíveis e consistentes.

### Quando `temperature > 0`

O modelo usa **sampling** com `temperature` e `top_p`, gerando respostas mais variadas e criativas.

## Chat Template

O tokenizador Qwen usa `apply_chat_template` para formatar as mensagens no padrão esperado pelo modelo:

```python
messages = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "mensagem atual"},
]
```
