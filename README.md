# zylora

> Serverless GPU functions — deploy ML models with a decorator, scale to zero, pay per GPU-second.

[![PyPI](https://img.shields.io/pypi/v/zylora)](https://pypi.org/project/zylora/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

## Quickstart

```bash
pip install zylora
```

### Define a GPU function

```python
import zylora

@zylora.fn(gpu="H100")
def embed(text: str) -> list[float]:
    from transformers import AutoModel
    model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
    return model.encode(text).tolist()
```

### Deploy

```bash
zy deploy
```

### Invoke

```python
# Local (development)
result = embed("hello world")

# Remote (deployed)
result = embed.remote("hello world")

# Batch
results = embed.map(["hello", "world", "foo"])

# Streaming (for LLMs)
for token in generate.stream("Tell me a joke"):
    print(token, end="", flush=True)

# Async
job = embed.remote_async("hello")
result = job.result()
```

### Client API (without decorator)

```python
from zylora import Zylora

zy = Zylora()
result = zy.invoke("embed", {"text": "hello"})
```

## Authentication

Set your API key via environment variable:

```bash
export ZYLORA_API_KEY="zy_live_..."
```

Or use the CLI:

```bash
zy login
```

## Configuration

### `@zylora.fn()` options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gpu` | `str` | *required* | GPU type: `t4`, `l4`, `rtx4090`, `a100_40gb`, `a100_80gb`, `l40s`, `h100`, `h200`, `b200`, `mi300x` |
| `name` | `str` | function name | Custom function name |
| `packages` | `list[str]` | `[]` | pip packages to install |
| `model` | `str` | `None` | Model to download during build |
| `timeout` | `int` | `300` | Execution timeout (seconds) |
| `min_instances` | `int` | `0` | Minimum warm instances |
| `max_instances` | `int` | `10` | Maximum concurrent instances |
| `concurrency` | `int` | `1` | Requests per instance |
| `image` | `str` | `None` | Custom base Docker image |
| `runtime` | `str` | `"python312"` | Python runtime version |
| `routing` | `str` | `"cost_optimized"` | Routing strategy |
| `visibility` | `str` | `"private"` | Function visibility |

## Documentation

Full documentation: [docs.zylora.dev](https://docs.zylora.dev)

## License

MIT
