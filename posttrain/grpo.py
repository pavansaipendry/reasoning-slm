"""GRPO from scratch (Act II) — the reasoning showpiece.

Group Relative Policy Optimization (DeepSeek): for each prompt, sample a group of G
completions, score each with a *verifiable* reward (posttrain/rewards.py), and use the
group-normalized reward as the advantage — no value network. Update the policy with a
PPO-style clipped objective plus a KL penalty to a frozen reference policy.

    for each step:
      for each prompt p (with gold answer):
        sample G completions from the policy
        r_i  = reward(completion_i, gold)                 # verifiable
        A_i  = (r_i - mean(r)) / (std(r) + eps)           # group-relative advantage
      ratio = exp(logp_policy - logp_old)
      L_pg  = -mean_tokens( min(ratio*A, clip(ratio,1-e,1+e)*A) )
      L_kl  = mean_tokens( exp(logp_ref - logp_policy) - (logp_ref - logp_policy) - 1 )
      loss  = L_pg + beta * L_kl

Example:
    python -m posttrain.grpo --ckpt checkpoints/sft/ckpt.pt --data-dir data/build_1b \
        --dataset arithmetic --group-size 8 --prompts-per-step 8 --steps 200 --out checkpoints/grpo
"""

from __future__ import annotations

import argparse
import os
import random

import torch

from .common import get_device, load_model, load_tokenizer_from, special_ids, token_logprobs
from .rewards import total_reward, numeric_reward
from . import data as ptdata


def first_eot_end(ids, prompt_len, eot_id):
    """Index just past the first eot in the completion (exclusive), or len(ids)."""
    for j in range(prompt_len, len(ids)):
        if ids[j] == eot_id:
            return j + 1
    return len(ids)


def build_dataset(name, n):
    if name == "arithmetic":
        return ptdata.arithmetic_rl(n)
    if name == "gsm8k":
        return ptdata.gsm8k_rl(n)
    raise ValueError(f"unknown dataset {name!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="policy init (SFT or base) checkpoint")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dataset", default="arithmetic", choices=["arithmetic", "gsm8k"])
    ap.add_argument("--n-prompts", type=int, default=2000)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--kl-coef", type=float, default=0.04)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default="checkpoints/grpo")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or get_device()
    os.makedirs(args.out, exist_ok=True)

    policy, mcfg_d = load_model(args.ckpt, device)
    ref, _ = load_model(args.ckpt, device)           # frozen reference
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    tok = load_tokenizer_from(args.data_dir)
    eot_id, pad_id = special_ids(tok)
    prompts = build_dataset(args.dataset, args.n_prompts)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95))
    rng = random.Random(0)
    G = args.group_size

    for step in range(args.steps):
        batch_prompts = rng.sample(prompts, args.prompts_per_step)
        seqs, masks, advs, rewards, accs = [], [], [], [], []

        # ---- rollouts: G completions per prompt, scored with verifiable rewards ----
        policy.eval()
        for rp in batch_prompts:
            pids = tok.encode(rp.prompt).ids
            plen = len(pids)
            inp = torch.tensor([pids] * G, device=device)
            out = policy.generate(inp, args.max_new_tokens, temperature=args.temperature,
                                  top_k=50, eot_id=eot_id)
            group_r = []
            group_rows = []
            for g in range(G):
                ids = out[g].tolist()
                end = first_eot_end(ids, plen, eot_id)
                comp_text = tok.decode(ids[plen:end])
                r = total_reward(comp_text, rp.gold)
                group_r.append(r)
                accs.append(numeric_reward(comp_text, rp.gold))
                mask = [0] * len(ids)
                for j in range(plen, end):
                    mask[j] = 1
                group_rows.append((ids, mask))
            mean_r = sum(group_r) / G
            var = sum((x - mean_r) ** 2 for x in group_r) / G
            std = var ** 0.5
            for (ids, mask), r in zip(group_rows, group_r):
                seqs.append(ids)
                masks.append(mask)
                advs.append((r - mean_r) / (std + 1e-4))
                rewards.append(r)

        # ---- pad into tensors ----
        T = max(len(s) for s in seqs)
        X = torch.full((len(seqs), T), pad_id, dtype=torch.long)
        M = torch.zeros((len(seqs), T), dtype=torch.float)
        for i, (s, m) in enumerate(zip(seqs, masks)):
            X[i, : len(s)] = torch.tensor(s)
            M[i, : len(m)] = torch.tensor(m, dtype=torch.float)
        X, M = X.to(device), M.to(device)
        A = torch.tensor(advs, device=device).unsqueeze(1)      # (B,1)
        mask = M[:, 1:]                                          # align to token_logprobs

        policy.train()
        with torch.no_grad():
            old_logp = token_logprobs(policy, X)
            ref_logp = token_logprobs(ref, X)

        last = {}
        for _ in range(args.ppo_epochs):
            logp = token_logprobs(policy, X)
            ratio = torch.exp(logp - old_logp)
            unclipped = ratio * A
            clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * A
            pg = -torch.min(unclipped, clipped)
            kl = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1.0
            denom = mask.sum().clamp(min=1.0)
            pg_loss = (pg * mask).sum() / denom
            kl_loss = (kl * mask).sum() / denom
            loss = pg_loss + args.kl_coef * kl_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            last = {"loss": loss.item(), "pg": pg_loss.item(), "kl": kl_loss.item()}

        mean_reward = sum(rewards) / len(rewards)
        acc = sum(accs) / len(accs)
        print(f"step {step:4d} | reward {mean_reward:.3f} | acc {acc:.3f} | "
              f"pg {last['pg']:+.4f} | kl {last['kl']:.4f} | loss {last['loss']:+.4f}", flush=True)

        if step % 25 == 0 or step == args.steps - 1:
            torch.save({"model": policy.state_dict(), "config": mcfg_d, "step": step},
                       os.path.join(args.out, "ckpt.pt"))

    torch.save({"model": policy.state_dict(), "config": mcfg_d, "step": args.steps},
               os.path.join(args.out, "ckpt.pt"))
    print(f"done. GRPO checkpoint -> {os.path.join(args.out, 'ckpt.pt')}")


if __name__ == "__main__":
    main()
