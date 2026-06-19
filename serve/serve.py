"""Lightweight inference server for the trained model (Act III).

Loads a checkpoint once and serves generation over HTTP — no framework dependency
(stdlib http.server). Exposes a simple `/generate` endpoint and an OpenAI-style
`/v1/chat/completions` so existing clients can talk to it. Reports tokens/sec.

(For high-throughput production serving, the model would be loaded into the mini-vllm
engine with PagedAttention + continuous batching; this server is the demo path.)

    python -m serve.serve --ckpt checkpoints/grpo_hard/ckpt.pt --data-dir data/build_1b --port 8000

    curl -s localhost:8000/generate -d '{"prompt":"What is 4 7 + 3 8?"}'
    curl -s localhost:8000/v1/chat/completions \
         -d '{"messages":[{"role":"user","content":"What is 4 7 + 3 8?"}]}'
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from posttrain.common import build_prompt, get_device, load_model, load_tokenizer_from, special_ids

STATE = {}  # model, tok, eot_id, device


def _generate(prompt_text: str, max_new_tokens: int, temperature: float) -> dict:
    tok, model, device, eot_id = STATE["tok"], STATE["model"], STATE["device"], STATE["eot"]
    ids = tok.encode(prompt_text).ids
    x = torch.tensor([ids], device=device)
    top_k = 1 if temperature <= 0 else 50
    t0 = time.time()
    out = model.generate(x, max_new_tokens, temperature=max(temperature, 1e-4), top_k=top_k, eot_id=eot_id)
    dt = time.time() - t0
    new = out[0].tolist()[len(ids):]
    return {"text": tok.decode(new).strip(), "tokens": len(new), "tokens_per_sec": round(len(new) / dt, 1)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "invalid json"})
        mnt = int(req.get("max_new_tokens", 64))
        temp = float(req.get("temperature", 0.0))

        if self.path == "/generate":
            r = _generate(build_prompt(req.get("prompt", "")), mnt, temp)
            return self._send(200, r)

        if self.path == "/v1/chat/completions":
            msgs = req.get("messages", [])
            user = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")
            r = _generate(build_prompt(user), mnt, temp)
            return self._send(200, {
                "object": "chat.completion",
                "model": "reasoning-slm",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": r["text"]},
                             "finish_reason": "stop"}],
                "usage": {"completion_tokens": r["tokens"]},
                "tokens_per_sec": r["tokens_per_sec"],
            })

        self._send(404, {"error": "unknown endpoint"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or get_device()
    model, _ = load_model(args.ckpt, device)
    model.eval()
    tok = load_tokenizer_from(args.data_dir)
    eot_id, _ = special_ids(tok)
    STATE.update(model=model, tok=tok, device=device, eot=eot_id)

    print(f"serving {args.ckpt} on :{args.port}  (device={device})")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
