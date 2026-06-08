"""Offline tests for the Act I pipeline and the verifiable rewards.

No network, no GPU. Run with:  python -m pytest tests/ -q   (or plain `python tests/test_pipeline.py`)
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dedup import MinHashDedup, _optimal_bands
from data.filters import FilterConfig, keep_document
from data.sources import Document, sample_documents
from posttrain.rewards import extract_answer, numeric_reward, format_reward, total_reward


def test_filters_keep_good_reject_junk():
    cfg = FilterConfig()
    docs = sample_documents()
    assert keep_document(docs[0], cfg) is None            # good prose kept
    assert keep_document(docs[4], cfg) is None            # good code kept
    assert keep_document(docs[5], cfg) == "too_short"     # "too short"
    # the all-'#' doc is one giant 600-char "word" -> tripped before symbol/alpha checks
    assert keep_document(docs[6], cfg) in {"mean_word_len_high", "symbol_ratio_high", "alpha_fraction_low"}
    assert keep_document(docs[8], cfg) == "mean_word_len_low"


def test_minhash_catches_near_and_exact_dups():
    d = MinHashDedup(num_perm=64, threshold=0.6, shingle_size=3)
    docs = sample_documents()
    assert d.is_duplicate(docs[0].text) is False          # first occurrence kept
    assert d.is_duplicate(docs[1].text) is True           # near-duplicate caught
    assert d.is_duplicate(docs[2].text) is True           # exact duplicate caught
    assert d.is_duplicate(docs[3].text) is False          # unrelated doc kept


def test_minhash_signature_estimates_jaccard():
    d = MinHashDedup(num_perm=256, threshold=0.5, shingle_size=2)
    a = d.signature("the quick brown fox jumps over the lazy dog")
    b = d.signature("the quick brown fox jumps over the lazy dog")
    assert d._estimated_jaccard(a, b) == 1.0              # identical -> 1.0
    c = d.signature("completely different words appear here only")
    assert d._estimated_jaccard(a, c) < 0.2


def test_optimal_bands_divides_num_perm():
    bands, rows = _optimal_bands(128, 0.8)
    assert bands * rows == 128


def test_rewards_are_verifiable():
    good = "<think>8 times 3 is 24</think> <answer>24</answer>"
    assert extract_answer(good) == "24"
    assert numeric_reward(good, "24") == 1.0
    assert numeric_reward(good, "25") == 0.0
    assert format_reward(good) == 1.0
    assert numeric_reward("no answer here", "24") == 0.0
    assert total_reward(good, "24") > total_reward(good, "25")


def test_full_sample_pipeline_produces_bins():
    from data.pipeline import run, _SAMPLE_CONFIG
    from data.pack import load_split

    with tempfile.TemporaryDirectory() as out:
        report = run(_SAMPLE_CONFIG, out, sample=True, limit=None)
        manifest = report["manifest"]
        assert manifest["train_tokens"] > 0
        assert os.path.exists(os.path.join(out, "tokenizer.json"))
        # near-dup + exact dup of doc 0 should have been removed
        assert report["stats"]["dedup"]["duplicates_removed"] >= 2
        arr = load_split(out, "train")
        assert len(arr) == manifest["train_tokens"]
        assert int(arr[-1]) == manifest["eot_id"]          # docs end with <eot>


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
