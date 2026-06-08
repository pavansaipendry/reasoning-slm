"""GRPO from scratch (Act II) — the showpiece. SCAFFOLD.

GRPO (Group Relative Policy Optimization, DeepSeek): for each prompt, sample a group
of G completions, score each with a *verifiable* reward (posttrain/rewards.py), and
use the group-normalized reward as the advantage — no value network. Update the policy
with a PPO-style clipped objective plus a KL penalty to a frozen reference.

Implementing this by hand (not just calling TRL) is the differentiator, given the
kernel/infra background. Sketch of the loop:

    for step:
        prompts = sample_batch()
        for p in prompts:
            comps = policy.generate(p, n=G, temperature=...)
            rewards = [reward_fn(c, gold[p]) for c in comps]
            adv = (rewards - mean(rewards)) / (std(rewards) + eps)   # group-relative
        logp  = policy.logprobs(comps)
        logp0 = ref.logprobs(comps)            # frozen reference
        ratio = exp(logp - logp_old)
        loss  = -min(ratio*adv, clip(ratio,1-eps,1+eps)*adv) + beta * KL(logp || logp0)
        loss.backward(); opt.step()

Parallel track: run the same loop on Qwen2.5-0.5B for a demo that actually pops.
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--group-size", type=int, default=8)
    ap.parse_args()
    raise NotImplementedError("GRPO stage lands after SFT. Rewards are ready in rewards.py.")


if __name__ == "__main__":
    main()
