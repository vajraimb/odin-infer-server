# 测试 odin-infer-server

启动 server 后用 `curl`（或任意 Ollama 兼容客户端）调用。接口与 Ollama 完全兼容。

## 1. 启动（后台）

```sh
cd /path/to/odin-infer-server

# Qwen3.5 / Ornith（Metal + repetition penalty + 2048 上下文）
# 端口用 11445 避开 Ollama 的 11434
./odin-infer-server /path/to/ornith-1.0-9b-Q4_K_M.gguf \
  --port 11445 -g 1 -c 2048 -x 1.15 > /tmp/srv.log 2>&1 &

# 等 5~10 秒加载完，看到 "listening on ..." 即就绪
sleep 8 && cat /tmp/srv.log
```

> `-g 1` 开 Metal，`-x 1.15` 开 repetition penalty（server 级，所有请求生效）。
> 架构由 GGUF 的 `general.architecture` 自动识别，Qwen3 旧模型同理：
>
> ```sh
> ./odin-infer-server Qwen3-0.6B-Q4_K_M.gguf --port 11446 -g 1
> ```

## 2. 调用接口

```sh
BASE=http://127.0.0.1:11445

# 版本 / 健康 / 已加载模型
curl -s $BASE/api/version
curl -s $BASE/health
curl -s $BASE/api/tags

# /api/generate —— 单轮 prompt（非流式）
curl -s $BASE/api/generate -d '{
  "model":"ornith",
  "prompt":"What is 2+2? One word.",
  "stream":false,
  "options":{"temperature":0}
}'

# /api/chat —— 多轮消息（非流式）
curl -s $BASE/api/chat -d '{
  "model":"ornith",
  "messages":[{"role":"user","content":"你擅长做什么？一句话。"}],
  "stream":false,
  "options":{"temperature":0.6,"top_p":0.95}
}'

# /api/chat —— 流式（NDJSON，一边生成一边吐）
curl -sN $BASE/api/chat -d '{
  "model":"ornith",
  "messages":[{"role":"user","content":"用三句话解释什么是混合注意力。"}],
  "stream":true
}'
```

非流式返回形如：

```json
{"model":"ornith-1.0-9b-Q4_K_M.gguf","created_at":"2026-06-30T03:26:44Z","message":{"role":"assistant","content":"Four"},"done":true,"prompt_eval_count":22,"eval_count":1}
```

## 3. 停止 server

```sh
pkill -f odin-infer-server
```

## 要点

- `options` 支持 `temperature` / `top_p` / `num_predict`；repetition penalty 是启动时 `-x` 设的 server 级参数。
- 任意 Ollama 客户端都能连 —— 把 `base_url` 指到 `http://127.0.0.1:11445` 即可。
- 跑 Qwen3 旧模型不用改代码，server 自动分派，换个端口避免和 Ornith 实例冲突就行。
