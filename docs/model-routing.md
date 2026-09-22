# Adaptive model routing (fork-specific)

This is a fork-specific addition to this `superpowers` checkout. It is **not**
part of upstream `obra/superpowers`, and per this repo's own `AGENTS.md`
("Fork-specific changes" / "Third-party dependencies"), it is not intended to
ever be upstreamed there — this document and the code under `router/` exist
only for this fork.

The full design is in [`docs/router/spec.md`](router/spec.md) (kept verbatim
as the reference document; code comments throughout `router/` cite its
section numbers). This document tracks what's actually built versus what
that spec describes, and records what's been verified against real sources
versus what remains an open assumption.

## What's implemented

Corresponds to spec sections 5–10, 22 (dry-run only), 29, and the "Phase 0–1"
scope from section 51/54:

- `router/schemas.py` — `TaskEnvelope`, `RouteDecision`, `ClassifierResult`,
  `AttemptRecord`, `VerificationResult`, `CommandResult`.
- `router/config.py` — YAML loading for `routing.yaml` with `${VAR}` /
  `${VAR:-default}` interpolation.
- `router/policy.py` — the deterministic tier-selection policy: base
  proposal, hard floors, ceilings (`max_tier`, `local_only`), the
  strong-verification speculative downgrade, the ambiguity upgrade, and the
  low-confidence safety net.
- `router/classifier.py` — `LayaClassifier` (optional dependency) and
  `DeterministicFallbackClassifier`, behind a shared `Classifier` interface.
- `router/cli.py` / `scripts/route-task` — `route-task --dry-run` only.
- `tests/router/` — 36 tests, all mocking Laya (no model download, no
  network call, per the Phase 1 exit criterion in spec section 51).

## What's explicitly NOT implemented yet

- The model broker and provider adapters (spec section 11–12): no code
  selects a concrete provider/model. `route-task` refuses to run without
  `--dry-run` for exactly this reason.
- SDD dispatch integration (spec section 18): `skills/subagent-driven-development/`
  has not been touched. No subagent dispatch anywhere in this repo goes
  through the router yet.
- Verification/escalation execution (spec sections 14–16), telemetry (spec
  sections 23–24), the static enforcement test (spec section 36), and
  everything else from Phase 2 onward.

Do not assume any of the above exists because the spec describes it in
detail — it doesn't yet.

## Laya: verified facts vs. open questions

Before writing `router/classifier.py`, the actual `laya` package (0.3.5) was
downloaded from PyPI and its source read directly — nothing here is inferred
from the spec's description of Laya or from Laya's own README claims.

**Verified from source** (`laya/router.py`, `laya/agent.py`):

- `laya` is a real PyPI package. Its `Requires-Dist` are `torch`,
  `transformers`, `safetensors`, `huggingface_hub`, `numpy` — a genuinely
  heavy ML stack, which is why it's an optional extra
  (`pip install '.[routing]'`) and never a core dependency.
- `Agent.system_one` / `Router.predict(state, questions, model=...)`
  mechanically accepts an arbitrary custom `questions` dict of type
  `choice` / `score` / `noul`, exactly the shape `ROUTING_QUESTIONS` uses.
  This was confirmed by reading `Agent._to_internal` and `Agent.system_one`'s
  actual return statement, not assumed from the spec.
- Laya ships three checkpoints: `english` (general-purpose default,
  ModernBERT-large, 421M params), `multilingual` (mmBERT-base), and
  `typed-decisions` (fine-tuned on four fixed, unrelated workflows:
  agent-trace observability, customer service, invoice processing, security
  incidents). `Router.route()`'s auto-task-detection only matches those four
  fixed workflows by exact question-id-set — our custom questions never
  match them, so `auto_task_detection=False` (as the spec's own sample code
  already sets) is correct, and we bypass `route()` entirely and call
  `predict(..., model="english")` explicitly.

**Not verified, and not verifiable without running an eval**: whether the
`english` checkpoint's judgments on software-engineering task-classification
questions (task type, complexity, cross-file reasoning, architectural
judgment, ambiguity, debugging reasoning) are *meaningful* rather than just
mechanically well-formed. Laya's own published benchmark data covers
customer-support intent classification, XNLI, and the four fixed
typed-decisions workflows — nothing about source code or software-engineering
task classification. This is a real, currently-open question, not something
papered over with a plausible-sounding default. It's the reason:

- `DeterministicFallbackClassifier` exists as a real fallback, not a stub.
- The low-confidence policy in `policy.py` treats classifier probabilities as
  a *proposal*, never as ground truth, and structural floors override it
  regardless of what the classifier says.
- Nothing in this codebase treats a Laya classification as evidence of task
  difficulty without corroborating structural signals (file count, explicit
  `metadata` flags for final-review/security/concurrency).

Resolving this open question requires telemetry (spec section 23+, not yet
built) correlating Laya's classifications against observed task outcomes —
i.e. exactly the Phase 6–7 work the spec itself defers.

## A few places this implementation filled gaps the spec left open

The spec (2487 lines, 60 sections) has some internal inconsistencies and
undefined types. Rather than silently picking an interpretation, each gap is
documented at the point it's filled:

- `TaskEnvelope.previous_failures` is `list[str]`: the spec references a
  `FailureEvidence` type in section 5.1 but never defines it anywhere.
- Combining Laya's seven independent per-question confidences into one
  `classifier_confidence` scalar isn't specified; `LayaClassifier._normalize`
  takes the minimum (conservative: as trustworthy as its weakest answer).
- Security/concurrency hard floors require an explicit
  `TaskEnvelope.metadata` flag (`security_sensitive`, `concurrency_sensitive`)
  rather than keyword-matching task text — guessing sensitivity from free
  text is exactly the kind of unreliable inference this router should avoid.
- Section 28's example config and section 10's example config nest
  confidence-related fields differently; `router/config.py` standardizes on
  section 28's location (`routing.confidence.*`).

## Running it

```bash
pip install -e .[dev]          # pyyaml + pytest, no Laya
python3 -m pytest tests/router/ -v

# Optional: install Laya for real classification
pip install -e .[routing]

scripts/route-task --input task.json --config config/routing.example.yaml --dry-run
```

`task.json` is a JSON-serialized `TaskEnvelope` (see `router/schemas.py` for
fields). `--config` defaults to `config/routing.example.yaml` if omitted.
