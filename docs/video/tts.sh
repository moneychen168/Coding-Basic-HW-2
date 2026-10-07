#!/bin/bash
# 登入節點執行（網路 I/O）。用法: bash tts.sh [投影片編號，省略=全部]
cd "$(dirname "$0")" && UV_CACHE_DIR=/tmp/uv-$USER .venv/bin/python tts.py "$@"
