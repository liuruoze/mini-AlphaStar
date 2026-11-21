#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""极简 OpenAI 兼容推理服务（单实例、全局锁串行生成），给多个并行对局共用。

## 端点
  POST /v1/chat/completions  单条；返回 {"choices":[{"message":{"content":...}}]}
  POST /v1/batch             N 条；返回 {"texts":[...], "latency_s":...}（左填充一次前向）
  GET  /health               调用数与平均耗时

## 为什么必须共用
        
8 个并行对局各自加载模型会 OOM。共用进程 + 全局 LOCK（单卡串行，避免显存与上下文竞争）。
注意：LOCK 意味着**客户端多线程并发不会加速**（实测 1.04x），要提吞吐只能走 /v1/batch。

## 两套部署（实测数，2026-11-21）

| 模型 | 权重 | 显存 | 5 条规则一次判定 | 
|---|---:|---|---:|
| Qwen2.5-7B  | 15 GB | 1 张 24 GB 卡 | 0.39 s（串行） |
| Qwen2.5-32B | 65 GB | 4 张卡，device_map=auto | 1.26 s（串行）/ 0.53 s（批量） |

## 仿真倍速与并发上限（2026-11-21 实测 15–17×，按 16× 估算）

`probe_sim_speed.py` 在同一套 SC2Env 上跑 no-op：200 步 47.2 步/秒（16.9×）、
400 步 43.1 步/秒（15.4×）。取保守值 **16×**。
注意这是**纯仿真上限**——真实评测每步还要跑一次策略网络前向，实际步速会低于此值。

`MAS_HYBRID_INTERVAL` 的单位是**游戏秒**，一次决策要问 5 条，所以：

```
单局需求 (条/秒) = 5 × v / I      v = 目标倍速，I = 间隔（游戏秒）
可支撑局数 N     = S × I / (5 × v)  S = 服务供给（串行≈4；批量 9.1 单局 5 条 / 11 跨局凑 20）
```

| 间隔 I | 目标 16×（满速） | 目标 8× | 目标 4× |
|---:|---:|---:|---:|
| 8  | 0.9 局 | 1.8 局 | 3.6 局 |
| 30 | 3.4 局 | 6.8 局 | 13.6 局 |
| 60 | 6.8 局 | 13.6 局 | 27.3 局 |

（上表按**每局一次批量（5 条）**供给 9.1 条/秒——实测 0.55 s/次，5/0.55 = 9.1。串行只有 4.0 条/秒，
每格除以 2.3，例如 I=30、16× 时只有 1.5 局。若做**跨局合并**把批量凑到 20，供给可到 11 条/秒，
上表乘以 1.2。）

⇒ 满速 16× 下：**批量勉强喂 1 局**（I=8，0.9 局），8 局并发要 I≈70（60 时只有 6.8 局）。
　要更快，只能加实例、上 vLLM，或做跨局合并。

早期按"1× 实时"算出的"14 局"是错的（差 4 倍）。

7B 与 32B **不能共存**在 4 卡机上（32B 独占四卡：24/18/18/23 GB）。

## 批量与串行的答案不完全相同（重要）

同一局面、同一提示词，批量 5 条与串行的答案在阈值边界上会不同，且**随批量大小变化**
（n=1 是、n=2 否、n=3 是、n=4/5 否）。这是 GEMM 数值路径差异，不是左填充的错（n=1 与串行一致）。
代价实测：串行 0.925 / 批量 0.912（120 题）。

⇒ **评测一律串行**；批量只用于实盘吞吐，且分数按批量口径单独记。

## 启动

  # 7B（单卡）
  CUDA_VISIBLE_DEVICES=0 ~/llm_env/bin/python experiments/hybrid/llm_server.py \
      --model-path ~/models/Qwen/Qwen2___5-7B-Instruct --port 8765

  # 32B（四卡自动切分）
  LLM_DEVICE_MAP=auto ~/llm_env/bin/python experiments/hybrid/llm_server.py \
      --model-path ~/models/Qwen/Qwen/Qwen2___5-32B-Instruct --port 8769

## 自检

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
