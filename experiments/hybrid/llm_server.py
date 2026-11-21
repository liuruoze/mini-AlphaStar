#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""极简 OpenAI 兼容推理服务（单实例、串行生成），给多个并行对局共用。

为什么需要：8 个并行对局若各自加载 7B（15 GB）会直接 OOM；共用一个进程即可。
串行生成意味着吞吐 ≈ 1/延迟；LLM 指挥官的建议间隔因此取 30 s 而不是 8 s。

启动:
  CUDA_VISIBLE_DEVICES=0 ~/llm_env/bin/python experiments/hybrid/llm_server.py \
      --model-path ~/models/Qwen/Qwen2___5-7B-Instruct --port 8765

自检:
  ~/llm_env/bin/python experiments/hybrid/llm_client.py --backend openai \
      --base-url http://127.0.0.1:8765/v1 --smoke
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

MODEL = None
LOCK = threading.Lock()
STATS = {"calls": 0, "total_s": 0.0}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, {"ok": True, "calls": STATS["calls"],
                             "avg_s": round(STATS["total_s"] / max(STATS["calls"], 1), 3)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._send(400, {"error": str(e)})
            return
        if self.path.rstrip("/").endswith("/batch"):
            # 一次请求算 N 条，返回顺序与输入一致
            batch = req.get("messages_batch", [])
            mt = int(req.get("max_tokens", 8))
            tp = float(req.get("temperature", 0.0))
            t0 = time.time()
            try:
                with LOCK:
                    texts = MODEL.chat_batch(batch, max_new_tokens=mt, temperature=tp)
            except Exception as e:
                self._send(500, {"error": "%s: %s" % (type(e).__name__, str(e)[:300])})
                return
            dt = time.time() - t0
            STATS["calls"] += len(batch)
            STATS["total_s"] += dt
            self._send(200, {"texts": texts, "latency_s": round(dt, 3)})
            return
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        msgs = req.get("messages", [])
        max_tokens = int(req.get("max_tokens", 128))
        temp = float(req.get("temperature", 0.2))
        t0 = time.time()
        with LOCK:                      # 单卡串行，避免显存与上下文竞争
            text = MODEL.chat(msgs, max_new_tokens=max_tokens, temperature=temp)
        dt = time.time() - t0
        STATS["calls"] += 1
        STATS["total_s"] += dt
        self._send(200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                      "finish_reason": "stop"}],
                         "usage": {"latency_s": round(dt, 3)}})

    def log_message(self, fmt, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.path.expanduser("~/models/Qwen/Qwen2___5-7B-Instruct"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()

    global MODEL
    import llm_client
    t0 = time.time()
    MODEL = llm_client.build("local", model_path=a.model_path)
    print("[server] model loaded in %.1fs (%s)" % (time.time() - t0, a.model_path), flush=True)

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print("[server] listening on http://%s:%d/v1" % (a.host, a.port), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
