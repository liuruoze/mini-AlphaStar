#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM 客户端：本地 transformers / OpenAI 兼容（vLLM、Ollama、任何 /v1）两种后端。

设计原则：
  - 默认本地（省钱）；只有显式设 LLM_BACKEND=openai 才走网络；
  - 只暴露一个 chat(messages) 接口，指挥官/复盘器共用；
  - 本地后端优先用 chat template（Qwen2.5-Instruct 自带），失败退回纯文本拼接。

用法:
  python experiments/hybrid/llm_client.py --smoke
  python experiments/hybrid/llm_client.py --backend openai --base-url http://127.0.0.1:8000/v1 --smoke
"""

import argparse
import json
import os
import sys
import time
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_LOCAL = os.environ.get("LLM_MODEL_PATH", os.path.expanduser("~/models/Qwen2.5-1.5B-Instruct"))


class LocalBackend:
    def __init__(self, model_path, device=None, max_new_tokens=256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True).to(self.device).eval()
        self.load_s = time.time() - t0
        self.max_new_tokens = max_new_tokens

    def chat(self, messages, max_new_tokens=None, temperature=0.2):
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tok([text], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **ids, max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                pad_token_id=self.tok.eos_token_id)
        gen = out[0][ids["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()


class OpenAIBackend:
    def __init__(self, base_url, model, api_key="EMPTY", timeout=60):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.key = api_key
        self.timeout = timeout
        self.load_s = 0.0

    def chat(self, messages, max_new_tokens=256, temperature=0.2):
        body = json.dumps({"model": self.model, "messages": messages,
                           "max_tokens": max_new_tokens, "temperature": temperature}).encode()
        req = urllib.request.Request(self.url, data=body, headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + self.key})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            obj = json.loads(resp.read().decode())
        return obj["choices"][0]["message"]["content"].strip()


def build(backend=None, model_path=None, base_url=None, model_name=None, max_new_tokens=256):
    backend = (backend or os.environ.get("LLM_BACKEND", "local")).lower()
    if backend == "openai":
        return OpenAIBackend(
            base_url or os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            model_name or os.environ.get("LLM_MODEL_NAME", "qwen2.5-1.5b-instruct"))
    return LocalBackend(model_path or DEFAULT_LOCAL, max_new_tokens=max_new_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if not args.smoke:
        ap.error("目前只支持 --smoke（其余用法见 module docstring）")

    llm = build(args.backend, args.model_path, args.base_url, args.model_name)
    msgs = [
        {"role": "system", "content": "你是星际争霸2神族的战术指挥官。只回答一行 JSON，不要解释。"},
        {"role": "user", "content": "当前：人口 30/31，矿 400，气 100，兵营 1，使徒 4，敌方未见。给出下一步建造指令。"},
    ]
    t0 = time.time()
    out = llm.chat(msgs, max_new_tokens=96)
    dt = time.time() - t0
    print("[backend] %s  load=%.1fs  infer=%.2fs" % (type(llm).__name__, llm.load_s, dt))
    print("[out] %s" % out[:300])


if __name__ == "__main__":
    main()
