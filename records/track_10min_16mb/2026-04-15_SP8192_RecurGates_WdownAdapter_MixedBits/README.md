# SP8192 + Recurrent Gates + W_down Adapter + Mixed Bits

Implementation scaffold for a record-track experiment based on the 2026-04-09 SP8192 SOTA stack, with three additional ablation targets:

- Depth-conditioned scalar gates on recurrent residual lanes.
- W_down-only tiny random-basis adapters on recurrent MLP down projections.
- Hessian-aware mixed-bit GPTQ allocation after the architecture is fixed.

No leaderboard score is claimed yet. Fill in this README and `submission.json` only after real 8xH100 runs.

## Default Stack

- SP8192 tokenizer and `fineweb10B_sp8192` data.
- 11 physical layers, 512 model width, 8 attention heads, 4 KV heads.
- 3-layer recurrence over layers 3-5, enabled at training fraction 0.35.
- Parallel residuals from layer 7 onward.
- QK-Gain default `5.25`, with intended sweep at `5.25` and `5.35`.
- Legal score-first TTT enabled by default.
- GPTQ with Hessian calibration and Brotli-compressed artifact.

## New Controls

```bash
DEPTH_GATES_ENABLED=1
DEPTH_GATE_MAX_DELTA=0.25
WDOWN_ADAPTER_RANK=4
WDOWN_ADAPTER_ALPHA=0.1
WDOWN_ADAPTER_SEED=4242
MIXED_BITS_ENABLED=1
MIXED_ARTIFACT_TARGET_BYTES=15980000
```

Depth gates initialize to no-op through `1 + delta * tanh(raw_gate)`, with `raw_gate = 0`.

The W_down adapter is only attached to `blocks.3-5.mlp.proj`. It uses deterministic non-persistent random bases plus tiny trainable scale parameters, so the basis is regenerated from the seed at load time and is not stored in the model artifact.

Mixed-bit GPTQ starts from int6 matrices and int8 embeddings, upgrades recurrent write-back matrices to int7, and uses late `mlp.fc` int5 fallback attempts if the compressed artifact is over the target budget.

## Suggested Ablations

```bash
# A0: SOTA parity
SEED=42 QK_GAIN_INIT=5.25 DEPTH_GATES_ENABLED=0 WDOWN_ADAPTER_RANK=0 MIXED_BITS_ENABLED=0 TTT_ENABLED=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

# A1: gates only
SEED=42 QK_GAIN_INIT=5.25 DEPTH_GATES_ENABLED=1 WDOWN_ADAPTER_RANK=0 MIXED_BITS_ENABLED=0 TTT_ENABLED=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

# A2: gates + W_down adapter
SEED=42 QK_GAIN_INIT=5.25 DEPTH_GATES_ENABLED=1 WDOWN_ADAPTER_RANK=4 MIXED_BITS_ENABLED=0 TTT_ENABLED=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

# A3: QK sweep
SEED=42 QK_GAIN_INIT=5.35 DEPTH_GATES_ENABLED=1 WDOWN_ADAPTER_RANK=4 MIXED_BITS_ENABLED=0 TTT_ENABLED=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

# A4: final mixed-bit artifact
SEED=42 QK_GAIN_INIT=<best> DEPTH_GATES_ENABLED=1 WDOWN_ADAPTER_RANK=4 MIXED_BITS_ENABLED=1 TTT_ENABLED=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Run final candidates on seeds `42`, `314`, and `999`. Record pre-quant, quantized, sliding-window, TTT BPB, artifact bytes, training time, and eval time.

## Data Setup

```bash
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
```

The script validates that `VOCAB_SIZE` matches the SentencePiece model before training.

## Compliance Notes

- TTT is score-first: each chunk is scored before any update on that chunk.
- No ETLB by default.
- No n-gram cache, SLOT, rescoring, or multi-pass selection.
- `train_gpt.py` is intentionally left as readable source for easier review and iteration. If this becomes a serious record candidate, re-check artifact bytes because readable source is larger than the compressed wrapper used by several top submissions.
