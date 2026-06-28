# odin-infer-server

Local HTTP inference service for **Qwen3** GGUF models, built on the
[odin-infer](https://github.com/vajraimb/odin-infer) library. Exposes an
**Ollama-compatible API** so other apps can call it over HTTP.

## Prerequisites

- [Odin](https://odin-lang.org/) compiler
- Clone `odin-infer` as a sibling directory:

```
Claw/
├── odin-infer/          # library
├── odin-infer-mac/      # CLI
└── odin-infer-server/   # this repo
```

## Build

```sh
./build.sh
```

## Run

```sh
./odin-infer-server Qwen3-0.6B-Q4_K_M.gguf --port 11435 -g 1
```

Default bind: `127.0.0.1:11435` (port chosen to avoid conflict with Ollama on 11434).

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--host` | Bind address | `127.0.0.1` |
| `--port` | HTTP port | `11435` |
| `--name` | Model name in API | GGUF filename |
| `-c` | Max context length | `4096` |
| `-g` | Metal GPU (0/1) | `1` |
| `-j` | Matmul threads | CPU cores |
| `-t` | Default temperature | `0.6` |
| `-p` | Default top-p | `0.95` |

## API (Ollama-compatible)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Server info |
| GET | `/health` | Health check |
| GET | `/api/tags` | List loaded model |
| GET | `/api/version` | Server version |
| POST | `/api/generate` | Text completion |
| POST | `/api/chat` | Chat completion |

### Generate

```sh
curl http://127.0.0.1:11435/api/generate -d '{
  "model": "qwen3",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```

### Chat

```sh
curl http://127.0.0.1:11435/api/chat -d '{
  "model": "qwen3",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}'
```

### Streaming

Set `"stream": true` for newline-delimited JSON (`application/x-ndjson`).

## Notes

- Single-model server: one GGUF loaded at startup.
- Requests are handled sequentially (one inference at a time).
- Qwen3 chat template is applied automatically for `/api/chat`.
- Thinking mode is disabled by default (matches CLI `-k 0`).

## Related repos

- [odin-infer](https://github.com/vajraimb/odin-infer) — inference library
- [odin-infer-mac](https://github.com/vajraimb/odin-infer-mac) — interactive CLI
