#!/usr/bin/env bash
# Build odin-infer-server against the odin-infer library (sibling directory).
set -euo pipefail
cd "$(dirname "$0")"
LIB="../odin-infer"
odin build . -out:odin-infer-server -o:speed -no-bounds-check -disable-assert -microarch:native \
  -collection:ggml=$LIB \
  -collection:infer=$LIB \
  -collection:tokenizer=$LIB \
  -collection:sampler=$LIB
echo "Built ./odin-infer-server ($(du -h odin-infer-server | cut -f1))"
