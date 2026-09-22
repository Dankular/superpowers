# CODEX.md — Superpowers Adaptive Model Router Fork

> Implementation specification for forking `obra/superpowers` and adding Laya-driven model classification, cost-aware dispatch, verification-based escalation, provider fallback, and routing telemetry.
>
> Target baseline: upstream `obra/superpowers` current main, September 2026.

---

## 0. Mission

Build a fork of **Superpowers** that preserves its software-development methodology while replacing prose-only model selection with an executable routing layer.

The fork must:

1. Classify every agent/subagent task before dispatch.
2. Start each task on the cheapest model tier that is plausibly capable of solving it.
3. Prefer local and free models where appropriate.
4. Verify outputs using objective evidence whenever possible.
5. Escalate automatically when a cheaper worker fails, stalls, lacks context, or produces unverifiable/low-confidence work.
6. Preserve useful artifacts from failed attempts and pass them to the next model.
7. Explicitly specify a model for every subagent dispatch so workers never silently inherit the expensive parent/session model.
8. Keep provider selection separate from task-complexity classification.
9. Record enough telemetry to train a Superpowers-specific router later.
10. Fail safely: routing may optimize cost, but it must not weaken correctness gates already present in Superpowers.

The core design principle is:

> **Laya chooses where a task starts. Evidence determines where it finishes.**

---

## 1. Upstream behaviour that MUST be preserved

Superpowers currently uses `subagent-driven-development` (SDD) to execute an implementation plan by dispatching a fresh implementer for each task, followed by task-scoped review and broad final review.

Its current model-selection guidance distinguishes roughly between:

- mechanical / isolated work → cheap model,
- integration / debugging / multi-file judgment → standard model,
- architecture / design / broad reasoning → strongest available model,
- fix-loop escalation → a stronger model after repeated failure.

SDD also requires the worker model to be specified explicitly because omission can make a subagent inherit the parent/session model.

Our fork MUST preserve the methodology and replace the **subjective selection mechanism**, not the methodology itself.

Do not remove:

- fresh isolated implementer per independent task,
- task brief generation,
- task-scoped review,
- fix/re-review loops,
- final whole-branch review,
- TDD and verification expectations,
- isolated context by default,
- Superpowers skill discovery behaviour.

---

## 2. Non-goals

Do NOT turn Superpowers into a monolithic inference server.

Do NOT hard-code OpenAI, Anthropic, OpenRouter, Ollama, llama.cpp, vLLM, or any other provider into skill files.

Do NOT use Laya itself to generate code.

Do NOT let classifier confidence alone decide whether an implementation is correct.

Do NOT replace unit tests, integration tests, linters, type checkers, build systems, task reviewers, or final reviewers with model confidence.

Do NOT automatically route every task to a frontier model merely because the parent/controller is frontier-capable.

Do NOT discard failed worker attempts if they contain useful investigation, diffs, compiler output, or test results.

---

## 3. System architecture

```text
                    ┌─────────────────────────┐
                    │      SUPERPOWERS        │
                    │ methodology + skills    │
                    └────────────┬────────────┘
                                 │ TaskEnvelope
                                 ▼
                    ┌─────────────────────────┐
                    │      TASK ROUTER        │
                    │ Laya + deterministic    │
                    │ policy + constraints    │
                    └────────────┬────────────┘
                                 │ RouteDecision
                                 ▼
                    ┌─────────────────────────┐
                    │      MODEL BROKER       │
                    │ tier → providers/models │
                    │ health / rate fallback  │
                    └──────┬─────────┬────────┘
                           │         │
                 ┌─────────┘         └──────────────┐
                 ▼                                  ▼
        local/free worker                    paid/frontier worker
                 │                                  │
                 └──────────────────┬───────────────┘
                                    ▼
                    ┌─────────────────────────┐
                    │      VERIFICATION       │
                    │ tests / build / lint    │
                    │ types / review / checks │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                  PASS                       FAIL
                    │                         │
                    ▼                         ▼
                 continue            evidence + escalation
                                              │
                                              └─────→ TASK ROUTER
```

The router chooses a **logical tier**. The broker chooses a **concrete provider/model**.

These are separate responsibilities.

---

## 4. Proposed repository layout

Add the following structure to the Superpowers fork:

```text
superpowers/
├── router/
│   ├── __init__.py
│   ├── classifier.py
│   ├── policy.py
│   ├── broker.py
│   ├── providers.py
│   ├── verification.py
│   ├── escalation.py
│   ├── telemetry.py
│   ├── schemas.py
│   ├── config.py
│   └── errors.py
│
├── config/
│   ├── models.example.yaml
│   └── routing.example.yaml
│
├── scripts/
│   ├── route-task
│   ├── dispatch-routed
│   ├── verify-task
│   └── routing-report
│
├── skills/
│   ├── model-routing/
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── routing-contract.md
│   │       └── escalation-contract.md
│   │
│   ├── subagent-driven-development/
│   │   ├── SKILL.md
│   │   ├── implementer-prompt.md
│   │   ├── spec-reviewer-prompt.md
│   │   └── code-quality-reviewer-prompt.md
│   │
│   └── ...existing skills...
│
├── tests/
│   ├── router/
│   │   ├── test_classifier.py
│   │   ├── test_policy.py
│   │   ├── test_broker.py
│   │   ├── test_escalation.py
│   │   ├── test_verification.py
│   │   └── test_telemetry.py
│   │
│   ├── integration/
│   │   ├── test_routed_sdd.py
│   │   ├── test_provider_fallback.py
│   │   ├── test_verified_escalation.py
│   │   └── test_no_implicit_model_dispatch.py
│   │
│   └── fixtures/
│
└── docs/
    ├── model-routing.md
    └── routing-dataset.md
```

Prefer Python for the routing subsystem because Laya exposes a Python API.

Keep scripts thin. Core behaviour belongs in `router/`, not shell scripts.

---

# 5. Core data contracts

## 5.1 `TaskEnvelope`

Every model dispatch MUST be normalized into one internal structure before routing.

```python
@dataclass
class TaskEnvelope:
    task_id: str
    role: str
    title: str
    description: str

    plan_text: str | None = None
    task_brief_path: str | None = None
    repository_root: str | None = None

    expected_files: list[str] = field(default_factory=list)
    expected_file_count: int | None = None
    languages: list[str] = field(default_factory=list)

    acceptance_criteria: list[str] = field(default_factory=list)
    verification_commands: list[str] = field(default_factory=list)

    previous_attempts: list["AttemptRecord"] = field(default_factory=list)
    previous_failures: list["FailureEvidence"] = field(default_factory=list)

    requires_web: bool = False
    requires_repo_search: bool = False
    requires_architecture: bool = False
    destructive_operations: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)
```

The router should see the task specification and useful structural metadata, but MUST NOT automatically dump the entire parent transcript into Laya or the worker.

---

## 5.2 `RouteDecision`

```python
@dataclass
class RouteDecision:
    task_id: str

    tier: str
    minimum_tier: str
    maximum_tier: str

    task_type: str
    complexity: int
    verification_strength: int

    cross_file_reasoning_probability: float
    architectural_judgment_probability: float
    ambiguity_probability: float
    debugging_probability: float

    classifier_confidence: float

    forced_by_policy: bool = False
    policy_reason: str | None = None
    classifier_raw: dict[str, Any] = field(default_factory=dict)
```

The caller MUST NOT need to understand Laya's output structure.

`classifier.py` translates Laya output into this stable application contract.

---

## 5.3 `AttemptRecord`

```python
@dataclass
class AttemptRecord:
    attempt: int
    tier: str
    provider: str
    model: str

    started_at: str
    finished_at: str

    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None
    cost_usd: float | None
    latency_ms: int | None

    status: str
    worker_report: str | None
    patch_ref: str | None

    verification: "VerificationResult | None"
```

---

## 5.4 `VerificationResult`

```python
@dataclass
class VerificationResult:
    passed: bool

    build: str = "not_run"
    tests: str = "not_run"
    lint: str = "not_run"
    typecheck: str = "not_run"
    spec_review: str = "not_run"
    quality_review: str = "not_run"

    commands: list["CommandResult"] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    retryable: bool = True
    suspected_context_failure: bool = False
    suspected_reasoning_failure: bool = False
```

---

# 6. Laya classifier

## 6.1 Dependency

Add Laya as an optional routing dependency.

The implementation should support:

```bash
pip install laya
```

Do not require all existing Superpowers users to install Laya unless they enable adaptive routing.

Suggested extras:

```toml
[project.optional-dependencies]
routing = ["laya>=0.3"]
```

Pin more tightly after integration testing.

---

## 6.2 Laya checkpoint usage

Laya itself has an internal `Router` for choosing among Laya checkpoints. That is separate from our **model broker**.

Use Laya's `Router` for classifier inference:

```python
from laya import Router

classifier_router = Router(
    preload=True,
    auto_task_detection=False,
)
```

For the initial implementation, explicitly select the appropriate Laya checkpoint for our custom schema rather than assuming typed-decisions workflow auto-detection will understand our new Superpowers-specific question set.

The classifier must be wrapped behind our own interface so Laya can later be replaced or fine-tuned without changing Superpowers skills.

---

## 6.3 Routing questions

Start with a custom question schema rather than routing only on a single vague difficulty score.

```python
ROUTING_QUESTIONS = {
    "task_type": {
        "type": "choice",
        "instructions": "Classify the primary software-engineering task.",
        "criteria": {
            "mechanical": "A deterministic or repetitive edit with little judgment.",
            "implementation": "Implement specified functionality.",
            "debugging": "Find and repair incorrect behaviour.",
            "refactor": "Restructure implementation while preserving behaviour.",
            "review": "Review an existing change or proposed implementation.",
            "architecture": "Make broad design, interface, or architectural decisions.",
            "research": "Discover or reconcile unfamiliar technical information.",
            "planning": "Break requirements into an implementation plan."
        },
    },

    "complexity": {
        "type": "score",
        "instructions": "Estimate reasoning complexity required for correct completion.",
        "criteria": [
            "trivial",
            "routine",
            "moderate",
            "difficult",
            "expert",
        ],
    },

    "verification_strength": {
        "type": "score",
        "instructions": "How objectively can success be verified after the worker finishes?",
        "criteria": [
            "strong: deterministic tests/build/checks cover the requested behaviour",
            "good: most important behaviour can be checked automatically",
            "partial: only some requirements can be checked automatically",
            "weak: correctness relies substantially on judgment or review",
        ],
    },

    "cross_file_reasoning": {
        "type": "noul",
        "instructions": "Does correct completion require coordinating behaviour across multiple files, modules, services, or APIs?",
    },

    "architectural_judgment": {
        "type": "noul",
        "instructions": "Does the task require architecture, API design, trade-off analysis, or broad system judgment?",
    },

    "ambiguity": {
        "type": "noul",
        "instructions": "Are the requirements materially ambiguous or underspecified?",
    },

    "debugging_reasoning": {
        "type": "noul",
        "instructions": "Does the task require diagnosing a cause that is not already known?",
    },
}
```

Do not feed repository secrets, environment variables, credentials, or unrelated source code to the classifier.

---

# 7. Logical model tiers

Model tiers MUST be names, not hard-coded concrete models.

Initial tiers:

```text
T0 = local_small
T1 = free_small
T2 = free_strong
T3 = economy
T4 = frontier
T5 = frontier_reasoning
```

Semantics:

### T0 — `local_small`

Use for:

- deterministic edits,
- code formatting,
- boilerplate,
- obvious one-file changes,
- transcription from a complete implementation plan,
- simple test generation when expected behaviour is explicit.

### T1 — `free_small`

Use for:

- routine implementation,
- isolated functions,
- simple bug fixes with a known cause,
- documentation and narrow tests.

### T2 — `free_strong`

Use for:

- ordinary coding work,
- moderate debugging,
- multi-file but well-scoped changes,
- integration with established project patterns,
- most first attempts where objective tests are strong.

### T3 — `economy`

Use for:

- tasks that exceed available free/local capacity,
- repeated failure at T2,
- higher-context integration,
- moderately ambiguous debugging.

### T4 — `frontier`

Use for:

- architecture,
- cross-cutting changes,
- difficult debugging,
- high-risk reviews,
- ambiguous requirements requiring significant judgment,
- repeated verified failure below this tier.

### T5 — `frontier_reasoning`

Reserve for:

- especially hard root-cause analysis,
- broad architectural reasoning,
- repeated failures at T4,
- difficult concurrency/security/correctness issues,
- final escalation when lower tiers cannot resolve the task.

The names MUST remain stable even if configured models change.

---

# 8. Model configuration

Create `config/models.example.yaml`:

```yaml
tiers:
  local_small:
    rank: 0
    providers:
      - provider: openai_compatible
        model: local-small
        base_url: ${LOCAL_MODEL_BASE_URL:-http://127.0.0.1:8080/v1}
        api_key_env: LOCAL_MODEL_API_KEY
        enabled: true
        priority: 10

  free_small:
    rank: 1
    providers:
      - provider: openrouter
        model: example/free-small
        api_key_env: OPENROUTER_API_KEY
        enabled: true
        priority: 10

  free_strong:
    rank: 2
    providers:
      - provider: openrouter
        model: example/free-coder
        api_key_env: OPENROUTER_API_KEY
        enabled: true
        priority: 10

  economy:
    rank: 3
    providers:
      - provider: openai_compatible
        model: economy-coder
        api_key_env: ECONOMY_API_KEY
        enabled: false
        priority: 10

  frontier:
    rank: 4
    providers:
      - provider: openai
        model: ${FRONTIER_MODEL}
        api_key_env: OPENAI_API_KEY
        enabled: true
        priority: 10

  frontier_reasoning:
    rank: 5
    providers:
      - provider: openai
        model: ${FRONTIER_REASONING_MODEL}
        api_key_env: OPENAI_API_KEY
        enabled: true
        priority: 10
```

The committed example MUST NOT contain real API keys.

Runtime credentials come from environment variables or the host harness's secret store.

---

# 9. Routing policy

The classifier proposes a starting point. Deterministic policy may override it.

Implement the policy in ordinary code so it is testable.

## 9.1 Hard floors

The following impose minimum tiers:

```text
architecture / API design                         >= T4
security-sensitive code review                    >= T4
final broad whole-branch review                   >= T4
difficult concurrency / distributed consistency   >= T4
unknown-cause debugging with broad scope           >= T2
multi-file implementation                          >= T2
review with non-trivial judgment                    >= T2
```

Make hard floors configurable.

Do not encode every future task category directly into SKILL.md.

---

## 9.2 Verification discount

Strong objective verification allows a task to start lower.

Example:

```text
Task complexity:       difficult
Architecture:          no
Requirements:          explicit
Automated verification: strong
```

This task may start at T2 instead of T4 because failed work can be cheaply detected.

By contrast:

```text
Task complexity:       moderate
Architecture:          yes
Verification:          weak
```

This should start at T4.

The policy should treat verification as a **risk reducer**, not proof that a weak model will succeed.

---

## 9.3 Initial policy implementation

Implement a deterministic first version roughly equivalent to:

```python
def choose_initial_tier(features: RoutingFeatures) -> str:
    if features.forced_minimum_tier:
        floor = features.forced_minimum_tier
    else:
        floor = "local_small"

    if features.architectural_judgment >= 0.70:
        proposed = "frontier"
    elif features.task_type == "architecture":
        proposed = "frontier"
    elif features.complexity >= 4:
        proposed = "frontier"
    elif features.complexity >= 3:
        proposed = "free_strong"
    elif features.cross_file_reasoning >= 0.65:
        proposed = "free_strong"
    elif features.task_type in {"mechanical"}:
        proposed = "local_small"
    else:
        proposed = "free_small"

    # Strong verification permits one-tier speculative downgrade,
    # but never below configured hard floors.
    if features.verification_strength == 0:
        proposed = one_tier_lower(proposed)

    # Ambiguity/debugging may push the task upward.
    if features.ambiguity >= 0.80 and features.verification_strength >= 2:
        proposed = one_tier_higher(proposed)

    return max_tier(proposed, floor)
```

Treat this as a baseline to test, not as an immutable algorithm.

---

# 10. Classification confidence policy

Laya probabilities MUST NOT be treated as absolute truth.

If classifier confidence is below a configured threshold:

1. do not automatically jump to the most expensive tier;
2. use conservative structural signals first;
3. prefer T2/T3 for ordinary coding when objective verification exists;
4. prefer T4 when ambiguity is high and objective verification is weak;
5. record low-confidence cases for later calibration/fine-tuning.

Configuration example:

```yaml
classifier:
  low_confidence_threshold: 0.58
  high_confidence_threshold: 0.82
  default_uncertain_tier: free_strong
```

---

# 11. Model broker

`broker.py` maps logical tiers to concrete providers.

Interface:

```python
class ModelBroker(Protocol):
    async def dispatch(
        self,
        task: TaskEnvelope,
        route: RouteDecision,
        worker_prompt: str,
    ) -> WorkerResult:
        ...
```

Provider adapters belong behind a common abstraction.

Minimum provider capability:

```python
class ProviderAdapter(Protocol):
    name: str

    async def healthcheck(self) -> ProviderHealth:
        ...

    async def run_agent(self, request: AgentRequest) -> AgentResponse:
        ...
```

Do not make provider adapters responsible for task classification.

---

# 12. Provider fallback

Within a tier, broker fallback should occur before model-tier escalation when the failure is infrastructural.

Examples of infrastructural failures:

- HTTP 429 / rate limit,
- provider unavailable,
- endpoint timeout,
- free quota exhausted,
- malformed provider response,
- temporary network error,
- model no longer exists.

Example:

```text
free_strong
   │
   ├─ provider A → 429
   ├─ provider B → timeout
   └─ provider C → success
```

This is NOT a reasoning escalation.

Only escalate the tier when the worker actually attempted the task and evidence indicates insufficient capability or repeated failure.

---

# 13. Worker context construction

A higher-tier retry MUST receive useful evidence from previous attempts.

Construct a compact escalation package:

```text
## Original task
...

## Acceptance criteria
...

## Previous attempt
Model: ...
Tier: ...
Summary: ...

## Patch/diff
...

## Verification failures
...

## Compiler/test output
...

## Known facts discovered by earlier worker
...

## What remains unresolved
...
```

Do not blindly include entire previous chain-of-thought transcripts.

Prefer:

- actual diff,
- tool output,
- test failures,
- compiler diagnostics,
- investigation summary,
- concrete hypotheses,
- files inspected,
- relevant plan context.

---

# 14. Verification pipeline

Verification determines whether the task is complete and supplies evidence for escalation.

The verification engine should support:

1. build command,
2. unit tests,
3. integration tests,
4. lint,
5. formatting checks,
6. type checks,
7. task acceptance checks,
8. spec compliance review,
9. code quality review,
10. optional custom project checks.

It MUST capture:

- command,
- exit code,
- stdout,
- stderr,
- duration,
- whether failure appears environmental or code-related.

---

## 14.1 Verification tiers

Classify evidence into:

### `STRONG`

Examples:

- deterministic unit/integration test suite,
- compiler errors or success,
- schema validation,
- golden-file comparison,
- exact acceptance test.

### `GOOD`

Examples:

- useful test coverage plus code review,
- lint/typecheck plus targeted runtime tests.

### `PARTIAL`

Examples:

- only static checks,
- some acceptance criteria automated.

### `WEAK`

Examples:

- architecture/design judgment,
- subjective UX review,
- maintainability assessment without objective oracle.

Weak verification should raise the starting model floor for judgment-heavy work.

---

# 15. Escalation policy

Escalation MUST be evidence-driven.

Do not simply retry the same prompt indefinitely.

## 15.1 Escalation reasons

Normalize failures into:

```text
INFRASTRUCTURE
MISSING_CONTEXT
REQUIREMENTS_AMBIGUOUS
BUILD_FAILURE
TEST_FAILURE
TYPE_FAILURE
LINT_FAILURE
SPEC_REVIEW_FAILURE
QUALITY_REVIEW_FAILURE
WORKER_BLOCKED
WORKER_NEEDS_CONTEXT
REPEATED_NO_PROGRESS
```

---

## 15.2 Response by failure type

### Infrastructure

Try another provider/model at the same tier.

### Missing context

Add context and retry the same tier once.

### Requirements ambiguous

Escalate to controller/human resolution if the task genuinely requires a product decision.

Do not spend higher model tiers guessing missing requirements.

### Build/test/type failure

Retry once at the same tier when the failure is local and actionable.

If the worker repeats the same error or shows no progress, escalate one tier.

### Spec review failure

Provide review findings to the implementer.

If the same class of spec failure repeats, escalate.

### Code quality review failure

Re-dispatch with concrete quality findings.

Escalate if judgment complexity appears above the current worker tier.

### Worker blocked

Inspect blocker reason:

- context problem → enrich context,
- reasoning problem → escalate,
- task too large → split,
- plan wrong → controller resolves plan.

This must remain aligned with Superpowers' existing BLOCKED/NEEDS_CONTEXT semantics.

---

# 16. Retry budget

Avoid unlimited speculative free-model loops.

Default budget:

```yaml
retry_policy:
  same_tier_reasoning_retries: 1
  same_tier_context_retries: 1
  provider_failovers_per_tier: 3
  maximum_total_attempts: 5
  maximum_frontier_attempts: 2
```

A cheap model taking six attempts is not necessarily cheaper than a capable model taking one.

Record turn count, latency, and token usage so later policy can optimize **total expected task cost**, not just per-token price.

---

# 17. Review routing

Reviewers are routed independently from implementers.

A task can therefore be:

```text
T1 implementer
   ↓
T2 reviewer
```

or:

```text
T2 implementer
   ↓
T4 reviewer
```

depending on diff risk.

Reviewer features should include:

- diff size,
- files touched,
- security relevance,
- concurrency relevance,
- public API changes,
- migration/data-loss risk,
- tests changed,
- generated code vs handwritten code,
- implementation tier,
- number of failed attempts.

Final whole-branch review MUST have a configurable minimum tier, default T4.

---

# 18. Integration with `subagent-driven-development`

Modify SDD so it no longer asks the controller to subjectively choose a model.

Replace the existing model-selection action with this contract:

```text
Before every implementer or reviewer dispatch:

1. Construct a TaskEnvelope.
2. Invoke `route-task`.
3. Read the returned logical tier and selected concrete model.
4. Dispatch the subagent with the model EXPLICITLY specified.
5. Record the attempt.
6. Verify.
7. Feed failure evidence back into the routing/escalation loop when necessary.
```

The skill should never contain a long list of current model names.

---

## 18.1 Proposed SDD skill wording

Add a section conceptually equivalent to:

```markdown
## Adaptive Model Routing

Do not choose worker models from intuition alone.

Before each implementer, task reviewer, re-reviewer, or final reviewer dispatch,
invoke the routing command with the complete task package.

The router returns a logical tier and concrete model selection.

Always pass the returned model explicitly to the subagent dispatch mechanism.
Never omit the model field.

When a worker fails verification or reports BLOCKED because the task exceeds
its reasoning capability, append the failure evidence and re-run routing.

Do not escalate for missing requirements. Resolve the missing requirement first.
Do not escalate tiers for provider outages. Use broker fallback within the tier.
```

Adapt syntax to each host runtime.

---

# 19. Route every subagent dispatch, not only SDD

Search the entire Superpowers repository for every code path / skill / prompt that dispatches a subagent.

For each dispatch site:

- determine its role,
- construct a `TaskEnvelope`,
- invoke routing,
- explicitly specify the worker model,
- add telemetry.

Add a repository test that fails if a known subagent dispatch template omits routed model selection.

The purpose is to eliminate the class of bug where non-SDD skills silently inherit the parent/session model.

---

# 20. Runtime adapter layer

Superpowers supports multiple coding-agent runtimes. Routing must not be coupled to only one.

Define a small runtime interface:

```python
class RuntimeAdapter(Protocol):
    name: str

    def supports_model_override(self) -> bool:
        ...

    async def spawn_worker(
        self,
        *,
        role: str,
        prompt: str,
        model: str,
        reasoning_effort: str | None = None,
        isolated: bool = True,
    ) -> WorkerHandle:
        ...
```

Initial priority:

1. Codex
2. Claude Code
3. OpenCode
4. generic OpenAI-compatible worker runner

Do not block the core routing package on every runtime being implemented.

---

# 21. Codex-specific integration

Preserve Superpowers' Codex isolated-subagent behaviour.

Requirements:

- use the runtime's multi-agent/collaboration support,
- prefer clean/isolated context for implementation workers,
- explicitly pass the concrete selected model where the Codex runtime permits it,
- do not rely on parent model inheritance,
- keep role prompts/task briefs as the primary worker context,
- preserve re-dispatch/follow-up semantics for fix rounds where appropriate.

If a Codex runtime/version cannot override the model for a particular spawn mode, detect that capability explicitly.

Do NOT silently claim routing occurred if the runtime ignored the requested model.

Emit a routing warning and apply the configured fallback strategy.

---

# 22. CLI

## `scripts/route-task`

Input:

```bash
scripts/route-task --input task-envelope.json
```

Output JSON only on stdout:

```json
{
  "task_id": "task-007",
  "tier": "free_strong",
  "provider": "openrouter",
  "model": "example/free-coder",
  "classifier": {
    "task_type": "implementation",
    "complexity": 2,
    "confidence": 0.81
  },
  "reason": "multi-file implementation with strong automated verification",
  "escalation_rank": 2
}
```

Human-readable diagnostics go to stderr.

Exit non-zero on unrecoverable configuration errors.

---

## `scripts/dispatch-routed`

Useful for runtimes that cannot directly call the Python module.

```bash
scripts/dispatch-routed \
  --task task-envelope.json \
  --prompt worker-prompt.md
```

It should:

1. route,
2. choose a provider,
3. invoke the configured adapter,
4. store attempt metadata,
5. return a structured result.

---

## `scripts/routing-report`

Examples:

```bash
scripts/routing-report --last 100
scripts/routing-report --by-tier
scripts/routing-report --escalations
scripts/routing-report --cost
scripts/routing-report --failures
```

---

# 23. Telemetry

Telemetry is a first-class requirement.

Store at minimum:

```text
task_id
repository_id_hash
role
task_type
classifier_features
classifier_confidence
initial_tier
selected_provider
selected_model
attempt_number
files_expected
files_changed
latency_ms
turn_count
prompt_tokens
completion_tokens
cached_tokens
estimated_cost
worker_status
build_result
test_result
lint_result
typecheck_result
spec_review_result
quality_review_result
escalation_reason
final_success
successful_tier
```

Do NOT store raw secrets.

Raw source code should be opt-in in telemetry.

Default telemetry should contain metadata and hashes rather than complete proprietary source files.

---

# 24. Local telemetry schema

Use JSONL initially for simplicity:

```text
.superpowers/router/events.jsonl
```

One event per routing/attempt/verification transition.

Example:

```json
{
  "event": "attempt_finished",
  "task_id": "task-007",
  "attempt": 2,
  "tier": "free_strong",
  "model": "...",
  "result": "test_failure",
  "latency_ms": 28441,
  "tests_passed": 41,
  "tests_failed": 2
}
```

Add SQLite later if querying JSONL becomes inconvenient.

---

# 25. Training dataset

The long-term dataset should answer:

> Given these task characteristics, what is the cheapest model tier that successfully completes the task within an acceptable retry/latency budget?

Generate training rows from completed tasks:

```json
{
  "features": {
    "role": "implementer",
    "task_type": "debugging",
    "file_count": 4,
    "verification_strength": "strong",
    "cross_file": true,
    "architecture": false,
    "initial_classifier_complexity": 3
  },
  "attempts": [
    {"tier": "free_small", "success": false},
    {"tier": "free_strong", "success": true}
  ],
  "label": "free_strong"
}
```

This later becomes training/fine-tuning data for a Superpowers-specific Laya classifier.

Avoid training directly on `initial_tier` chosen by the old policy; label from **observed successful outcomes**.

---

# 26. Cost objective

Do not optimize only per-token price.

The eventual objective should approximate:

```text
expected_total_cost =
    inference_cost
  + retry_cost
  + reviewer_cost
  + latency_penalty
  + failure_penalty
  + context_reconstruction_cost
```

A free model requiring four retries may be worse than a low-cost model succeeding in one turn.

Telemetry must make this measurable.

---

# 27. Cache

Cache classifier decisions for identical task envelopes when safe.

Cache key should include:

- normalized task description hash,
- role,
- expected file set/hash,
- routing policy version,
- classifier version,
- previous-failure state.

Do not reuse an initial classification after new failure evidence has materially changed the task state.

---

# 28. Configuration

Create `config/routing.example.yaml`:

```yaml
routing:
  enabled: true
  classifier: laya

  default_tier: free_strong
  final_review_min_tier: frontier

  confidence:
    low: 0.58
    high: 0.82

  speculative_downgrade:
    enabled: true
    max_levels: 1
    requires_strong_verification: true

  retry_policy:
    same_tier_reasoning_retries: 1
    same_tier_context_retries: 1
    provider_failovers_per_tier: 3
    maximum_total_attempts: 5
    maximum_frontier_attempts: 2

  telemetry:
    enabled: true
    path: .superpowers/router/events.jsonl
    include_source: false

  floors:
    architecture: frontier
    final_review: frontier
    security_review: frontier
    multi_file_implementation: free_strong
    nontrivial_review: free_strong
```

Environment override should be supported where practical.

---

# 29. Router failure mode

Routing is an optimization layer. Superpowers must remain usable if the classifier is unavailable.

If Laya cannot load:

1. log the failure,
2. use deterministic fallback policy based on structural signals,
3. never omit the explicit model selection,
4. continue if a configured broker tier is available.

Example fallback:

```text
complete code / one-file mechanical task     → T0/T1
ordinary implementation                      → T2
multi-file/debugging                          → T2/T3
architecture/design/final review              → T4
```

Do not route every classifier outage to T5.

---

# 30. Broker failure mode

If all providers in the selected tier are unavailable:

- if `allow_infrastructure_tier_fallback=true`, try the nearest configured tier according to policy;
- otherwise fail with a clear actionable error.

Do not classify provider outage as worker incompetence.

---

# 31. Security

The routing layer handles code and provider credentials. Apply these constraints:

- secrets only via environment/secret store,
- never write API keys to telemetry,
- redact known credential patterns from worker failure logs,
- do not send repository contents to external models unless the selected provider is permitted for that repository,
- support provider allow/deny policies per repository,
- support `local_only: true`,
- support `max_tier` to enforce spending/privacy limits,
- never escalate past a repository's allowed model/provider set.

Suggested repository policy:

```yaml
repository_policy:
  local_only: false
  max_tier: frontier
  allowed_providers:
    - local
    - openrouter
    - openai
  deny_external_paths:
    - secrets/
    - production-dumps/
```

---

# 32. Budget controls

Support optional budgets:

```yaml
budget:
  task_max_usd: 0.75
  session_max_usd: 10.00
  daily_max_usd: 25.00
  require_approval_above_tier: frontier_reasoning
```

Budget exhaustion must return an explicit state, not masquerade as model failure.

---

# 33. Observability

For every task, produce a concise route ledger:

```text
Task 7: implement parser cache
  route: free_strong
  classifier: implementation / moderate
  verification: strong
  worker: provider-x/model-y
  attempt 1: tests failed (2)
  retry: same tier, failure context added
  attempt 2: passed
  reviewer: free_strong
  result: accepted
```

This should be available to the controller and optionally printed to users without overwhelming normal Superpowers narration.

---

# 34. Testing strategy

Follow Superpowers' own skill-development/testing discipline.

The routing feature needs both ordinary unit tests and behavioural agent evaluations.

---

## 34.1 Unit tests

### Classifier

Test:

- Laya output normalization,
- missing fields,
- low-confidence result,
- malformed classifier response,
- classifier unavailable fallback,
- version metadata.

### Policy

Test deterministic cases:

```text
one-file complete spec + tests         → cheap tier
multi-file integration                 → >= free_strong
architecture                           → >= frontier
final review                           → >= frontier
strong verification speculative drop  → at most one level
weak verification                     → no unsafe downgrade
configured max_tier                    → never exceeded
configured floor                       → never undercut
```

### Broker

Test:

- first provider success,
- 429 fallback,
- timeout fallback,
- all providers unavailable,
- disabled provider ignored,
- policy denylist,
- tier selection independent of provider selection.

### Escalation

Test:

- infra failure does not increase tier,
- missing context retries same tier,
- repeated test failure increases tier,
- no-progress loop increases tier,
- ambiguity does not trigger expensive guessing,
- max-attempt cap enforced.

### Telemetry

Test:

- no credentials stored,
- events append correctly,
- successful tier derived correctly,
- source text omitted by default.

---

# 35. Behavioural/evaluation scenarios

Create pressure tests where the controller is tempted to ignore routing.

Examples:

### Scenario A — parent is expensive

Parent session uses a frontier model.

Task is a single-line deterministic change.

Expected:

- router selects T0/T1,
- subagent dispatch explicitly specifies cheaper model,
- parent model is not inherited.

### Scenario B — free worker fails

Task appears routine but free worker creates a failing patch.

Expected:

- verification catches failure,
- failure evidence is preserved,
- router/escalator advances tier,
- next worker sees previous failure evidence.

### Scenario C — provider outage

Free provider returns 429.

Expected:

- another provider at same tier is attempted,
- no reasoning escalation occurs solely because of 429.

### Scenario D — architecture task

Task requests a broad redesign with weak automated verification.

Expected:

- no speculative T0/T1 worker,
- configured architecture floor applies.

### Scenario E — ambiguous requirement

Worker cannot determine intended product behaviour.

Expected:

- do not keep escalating models hoping one will guess correctly,
- surface requirement ambiguity to controller/user.

### Scenario F — false economy

Free model repeatedly burns turns.

Expected:

- retry budget cuts it off,
- task escalates,
- telemetry records turn count and total cost.

---

# 36. Static enforcement test

Add a test that scans known Superpowers subagent dispatch prompts/configuration for model-selection compliance.

Goal:

> No subagent dispatch introduced in the repository may silently inherit a parent model unless explicitly documented as intentional.

If runtime syntax differs, use an allowlisted adapter mapping rather than a naive regex across every file.

---

# 37. Migration of existing model-selection prose

Do not simply delete the useful heuristics currently documented by Superpowers.

Move them into:

- `routing.example.yaml` floors/defaults,
- `policy.py`,
- test fixtures,
- model-routing documentation.

The SDD skill should become shorter and procedural:

```text
construct task → route → dispatch explicit model → verify → escalate if evidence requires
```

This reduces prompt interpretation drift.

---

# 38. Versioning

Record versions with every routing decision:

```text
router_policy_version
classifier_name
classifier_version
laya_version
model_config_version
runtime_adapter_version
```

Without versioned telemetry, routing data becomes difficult to compare after policy changes.

---

# 39. Proposed Python API

Target public API:

```python
from superpowers_router import RoutingEngine

engine = RoutingEngine.from_default_config()

decision = engine.route(task)

result = await engine.execute(
    task,
    worker_prompt=prompt,
)
```

Higher-level orchestration:

```python
result = await engine.execute_until_verified(
    task=task,
    worker_prompt=prompt,
    verifier=project_verifier,
)
```

Do not require skill markdown to parse raw Laya outputs.

---

# 40. Execution state machine

Implement an explicit state machine rather than scattered retry conditionals.

```text
NEW
 ↓
CLASSIFYING
 ↓
ROUTED
 ↓
DISPATCHING
 ↓
RUNNING
 ↓
VERIFYING
 ├──────── PASS ────────→ REVIEWING ──→ COMPLETE
 │
 └──────── FAIL
           ↓
      ANALYZE_FAILURE
       ├─ context ──────→ ENRICH_CONTEXT ─→ ROUTED
       ├─ infra ────────→ PROVIDER_FAILOVER → DISPATCHING
       ├─ reasoning ────→ ESCALATE_TIER ───→ ROUTED
       ├─ bad plan ─────→ CONTROLLER_DECISION
       └─ budget/cap ───→ STOPPED
```

State transitions should be logged.

---

# 41. Failure analyzer

Start rule-based.

Inputs:

- worker status,
- verifier outputs,
- test/build logs,
- previous attempts,
- provider error.

Do not initially spend another frontier-model call just to classify why a cheaper model failed.

A later version may use Laya or another classifier for failure categorization.

---

# 42. Task splitting

Escalation is not always the best solution.

If a task is too broad, the controller should be able to split it into independently verifiable subtasks.

Signals:

- many unrelated files,
- worker reports excessive scope,
- repeated partial completion,
- independent acceptance criteria,
- multiple systems/components involved.

Prefer:

```text
large T4 task
```

becoming:

```text
3 × T2 tasks + T4 integration review
```

when decomposition is valid.

This remains a controller/Superpowers decision, not an automatic classifier rewrite of the user's requirements.

---

# 43. Parallelism

Preserve Superpowers' safety rules around conflicting implementation work.

Routing does not imply that every cheap task should run simultaneously.

Parallel dispatch is allowed only when the existing methodology says tasks are independent.

The broker may enforce provider concurrency limits independently.

---

# 44. Provider health

Maintain transient health state:

```python
@dataclass
class ProviderHealth:
    available: bool
    rate_limited: bool
    cooldown_until: datetime | None
    recent_latency_ms: float | None
    recent_error_rate: float | None
```

Broker selection within a tier may consider:

- configured priority,
- provider availability,
- latency,
- remaining free quota when available,
- recent error rate,
- context-window requirement.

Do not use benchmark quality here; capability belongs in tier configuration/policy.

---

# 45. Context requirements

Models inside one tier may have different context windows.

The broker must reject models that cannot accommodate the task package.

Before dispatch estimate:

```text
system prompt
+ skill prompt
+ task brief
+ relevant code/context
+ previous failure evidence
+ output reserve
```

Then choose only eligible models.

Do not truncate acceptance criteria merely to fit a cheaper model.

---

# 46. Artifact preservation

After each attempt store:

- worker report,
- changed-files list,
- git diff/commit reference,
- verification logs,
- review findings.

Use Git wherever possible as the canonical patch history.

When a higher-tier worker takes over, clearly identify whether it should:

- continue the current patch,
- revert and reimplement,
- only fix remaining failures.

Do not accidentally layer incompatible failed patches.

---

# 47. Git/worktree behaviour

Do not weaken Superpowers' existing worktree strategy.

Routing metadata should live outside source commits by default, e.g.:

```text
.superpowers/router/
```

Add telemetry paths to `.gitignore` unless users explicitly opt to commit them.

Router configuration examples may be committed; secrets and task telemetry should not.

---

# 48. User-facing controls

Support simple overrides:

```text
--routing off
--tier free_strong
--min-tier economy
--max-tier frontier
--local-only
--provider openrouter
--no-escalation
--budget 0.50
```

Explicit user constraints override classifier recommendations.

Do not silently violate a user's max tier/budget/provider restriction.

---

# 49. Dry-run mode

Add:

```bash
scripts/route-task --dry-run --input task.json
```

Output:

```text
Task: Implement cache invalidation
Classifier:
  type: debugging
  complexity: moderate
  cross-file: 0.72
  architecture: 0.18
  verification: strong

Policy:
  structural floor: free_strong
  verification discount: no change

Selected:
  tier: free_strong
  provider: openrouter
  model: ...
```

Dry run makes policy tuning practical before enabling automatic dispatch.

---

# 50. Routing explainability

Every decision should have a concise machine-readable reason.

Example:

```json
{
  "tier": "free_strong",
  "reason_codes": [
    "MULTI_FILE",
    "MODERATE_COMPLEXITY",
    "STRONG_VERIFICATION"
  ],
  "summary": "Multi-file implementation requires T2 floor; strong tests make a higher starting tier unnecessary."
}
```

Do not expose internal chain-of-thought.

Expose observable features and policy rules.

---

# 51. Phase plan

## Phase 0 — Baseline and tests

Before changing behaviour:

- fork Superpowers,
- record upstream revision,
- run existing tests/evals,
- inspect all subagent dispatch sites,
- add regression test demonstrating implicit model inheritance risk,
- add fixtures for representative task classes.

Exit criteria:

- baseline tests pass,
- all dispatch sites inventoried.

---

## Phase 1 — Router core

Implement:

- schemas,
- config loader,
- Laya classifier wrapper,
- deterministic fallback classifier,
- routing policy,
- unit tests,
- `route-task --dry-run`.

Do NOT modify SDD dispatch yet.

Exit criteria:

- task fixtures route deterministically,
- classifier outage fallback works,
- no provider calls are required to test routing.

---

## Phase 2 — Broker

Implement:

- logical tiers,
- provider adapters,
- provider health/fallback,
- context eligibility,
- secrets handling,
- broker unit tests.

Exit criteria:

- a synthetic worker request can be dispatched through at least one local/OpenAI-compatible adapter,
- same-tier failover works.

---

## Phase 3 — SDD integration

Modify:

- `skills/subagent-driven-development/SKILL.md`,
- implementer dispatch template,
- task reviewer dispatch,
- re-review dispatch,
- final reviewer dispatch.

Every dispatch must explicitly carry selected model information.

Exit criteria:

- SDD simple task uses low tier,
- architecture task respects floor,
- final review respects minimum tier,
- no SDD dispatch inherits model unintentionally.

---

## Phase 4 — Verification/escalation

Implement:

- VerificationResult,
- command capture,
- failure categorization,
- retry budget,
- evidence package,
- tier escalation,
- tests.

Exit criteria:

- failed low-tier coding task automatically reaches a higher tier,
- higher-tier worker receives prior evidence,
- provider outage does not cause capability escalation.

---

## Phase 5 — All dispatch sites

Audit and migrate every other Superpowers subagent dispatch.

Add static regression test.

Exit criteria:

- repository contains no unintended implicit-model subagent dispatch paths.

---

## Phase 6 — Telemetry

Implement:

- JSONL event log,
- report CLI,
- cost/token metadata,
- successful-tier derivation,
- privacy controls.

Exit criteria:

- every route/attempt/verification/escalation is represented in telemetry.

---

## Phase 7 — Policy tuning

Run a corpus of real coding tasks.

Measure:

- first-attempt success by tier,
- eventual success,
- average attempts,
- latency,
- total token usage,
- total cost,
- false-low routes,
- unnecessarily-high routes.

Tune deterministic policy before attempting custom model training.

---

## Phase 8 — Superpowers-specific Laya training

Only after enough telemetry exists:

1. derive observed cheapest-successful-tier labels,
2. remove/limit sensitive source content,
3. split by repository/project to reduce leakage,
4. train/calibrate a domain-specific routing model,
5. evaluate against deterministic baseline,
6. ship only if it improves cost/latency without unacceptable success regression.

Keep deterministic policy floors even after model fine-tuning.

---

# 52. Acceptance criteria

The feature is not complete until all of the following are true.

## Functional

- [ ] Every SDD implementer dispatch is routed.
- [ ] Every SDD reviewer dispatch is routed.
- [ ] Final review has a configurable minimum tier.
- [ ] Every routed dispatch specifies a concrete model explicitly.
- [ ] Laya classifier can be enabled/disabled.
- [ ] Router has deterministic fallback if Laya is unavailable.
- [ ] Logical tiers are provider-independent.
- [ ] Provider failure can fall back within the same tier.
- [ ] Verification failure can trigger capability escalation.
- [ ] Missing context does not automatically trigger expensive escalation.
- [ ] Failed attempt evidence is preserved for the next worker.
- [ ] Retry limits prevent infinite cheap-model loops.
- [ ] User/provider/budget ceilings are respected.

## Quality

- [ ] Existing Superpowers test suite remains green.
- [ ] Routing policy has deterministic unit tests.
- [ ] Behavioural evals show cheap tasks no longer inherit expensive session models.
- [ ] Architecture tasks do not get unsafe speculative downgrades.
- [ ] Provider 429 does not count as reasoning failure.
- [ ] Telemetry never contains API keys.

## Observability

- [ ] Every task records initial tier.
- [ ] Every attempt records actual provider/model.
- [ ] Every escalation records a reason.
- [ ] Successful tier can be derived.
- [ ] Cost/latency/turn-count metrics are queryable.

---

# 53. Definition of done for the first shippable version

The first shippable release does NOT need custom Laya fine-tuning.

It DOES need:

```text
Laya classification
+ deterministic policy
+ logical tiers
+ one or more provider adapters
+ explicit dispatch model selection
+ SDD integration
+ verification-driven escalation
+ retry limits
+ telemetry
+ tests
```

A good MVP demo is:

```text
1. Start Codex/Superpowers on a frontier controller model.
2. Give it an implementation plan containing:
   - trivial one-file task,
   - moderate multi-file task,
   - difficult architecture/review task.
3. Show that each receives a different explicit worker tier.
4. Force the cheap worker to fail a deterministic test.
5. Show the task automatically escalate.
6. Show the stronger worker receive previous test evidence.
7. Show successful completion and routing telemetry.
```

---

# 54. Recommended first implementation order for Codex

When implementing this specification, work in this order:

1. Inspect upstream and enumerate every dispatch surface.
2. Add tests proving the current implicit-model problem.
3. Add `router/schemas.py`.
4. Add config loading and logical tier model.
5. Add deterministic routing policy first.
6. Add Laya as a classifier adapter behind the stable interface.
7. Add `route-task --dry-run`.
8. Add provider/broker abstraction.
9. Integrate SDD implementer dispatch.
10. Integrate SDD reviewer dispatch.
11. Add verifier + escalation loop.
12. Add attempt evidence package.
13. Migrate remaining dispatch sites.
14. Add static enforcement test.
15. Add telemetry/reporting.
16. Run behavioural evals.
17. Tune policy.
18. Only then consider training a specialized classifier.

Do not begin by rewriting every skill.

Get the router contract and SDD path working end-to-end first.

---

# 55. Engineering rules for Codex while modifying this fork

1. Preserve upstream behaviour unless this specification explicitly changes it.
2. Prefer small, reviewable commits.
3. Add tests before behavioural changes where practical.
4. Do not hard-code current commercial model names in skills.
5. Do not commit credentials.
6. Keep policy deterministic and inspectable.
7. Keep provider logic out of Laya classifier code.
8. Keep Laya output parsing out of skills.
9. Use explicit structured contracts between classifier, policy, broker, verifier, and runtime adapter.
10. Never interpret a successful model API call as task success; verification decides success.
11. Never interpret a provider outage as evidence that a stronger model is needed.
12. Never let a model tier exceed explicit user/repository budget or privacy limits.
13. Never silently fall back to parent/session model inheritance.
14. Any unavoidable runtime limitation that prevents model override must be visible in logs and telemetry.
15. Preserve Superpowers' context-isolation principles.

---

# 56. Initial issues / work breakdown

Create implementation issues approximately as follows:

### ROUTER-001 — Routing schemas and config

Implement task/route/attempt/verification schemas and YAML config loading.

### ROUTER-002 — Deterministic policy engine

Implement tier ranks, floors, ceilings, verification discount, and confidence handling.

### ROUTER-003 — Laya classifier adapter

Add optional dependency, custom Superpowers question schema, output normalization, fallback behaviour.

### ROUTER-004 — Dry-run CLI

Implement `scripts/route-task` and human-readable explanation mode.

### BROKER-001 — Provider abstraction

Implement provider interface and first OpenAI-compatible adapter.

### BROKER-002 — Same-tier provider fallback

Implement health, 429/timeout handling, and provider selection.

### SDD-001 — Routed implementer dispatch

Replace prose model selection with route call + explicit model.

### SDD-002 — Routed task reviews

Route spec and code-quality reviewers independently.

### SDD-003 — Routed final review

Enforce configurable final-review floor.

### VERIFY-001 — Verification result model

Normalize build/test/lint/typecheck evidence.

### ESC-001 — Evidence-driven retry/escalation

Implement failure classes, same-tier retry, escalation, caps.

### ESC-002 — Failure evidence package

Pass prior patch/test/compiler evidence to stronger worker.

### AUDIT-001 — Migrate all subagent dispatch sites

Eliminate accidental parent-model inheritance outside SDD.

### AUDIT-002 — Static regression guard

Fail CI for unapproved dispatch paths lacking explicit routing/model selection.

### TELEMETRY-001 — JSONL routing ledger

Record decisions, attempts, verification, cost, latency, final tier.

### TELEMETRY-002 — Routing report CLI

Summarize tier usage, escalations, success, and cost.

### EVAL-001 — Behavioural routing eval suite

Add simple/medium/architecture/provider-outage/ambiguous-requirement pressure scenarios.

---

# 57. Future extensions

Do not block MVP on these, but keep the architecture compatible with them.

## Learned tier probability

Instead of outputting one tier, estimate:

```text
P(success | T0)
P(success | T1)
P(success | T2)
P(success | T3)
P(success | T4)
P(success | T5)
```

Then choose the cheapest tier satisfying a target success threshold.

Example:

```text
minimum tier with P(success) >= 0.75
```

This is ultimately better aligned with the real routing problem than a generic difficulty label.

## Dynamic provider economics

Allow the broker to ingest current provider quotas/pricing and choose among equivalent models inside a tier.

Do not let transient pricing redefine capability tiers automatically.

## Repository-specific routing profiles

Learn that some projects are easier/harder for particular model families because of language/framework/style.

## Role-specific classifiers

Separate policies for:

- implementer,
- debugger,
- spec reviewer,
- quality reviewer,
- researcher,
- planner.

## Speculative parallel attempts

For selected high-latency tasks, optionally launch two cheap independent workers and keep the first verified success.

This must be opt-in because it changes resource economics and may conflict with Superpowers' implementation isolation rules.

## Bandit policy

Use contextual-bandit style policy updates from real completion outcomes.

Only pursue this after deterministic routing telemetry is trustworthy.

---

# 58. Final architecture invariant

At all times the following separation must hold:

```text
Superpowers
    decides HOW software work should be performed

Laya classifier
    estimates WHAT KIND of reasoning the task requires

Routing policy
    decides WHICH capability tier is appropriate

Model broker
    decides WHICH concrete provider/model satisfies that tier

Worker
    performs the task

Verifier/reviewers
    decide WHETHER the result is acceptable

Escalation engine
    decides WHAT to do after failure
```

No single component should silently absorb all six responsibilities.

That separation is the main architectural requirement of this fork.

---

# 59. Reference baseline

Implementation should be checked against the actual current upstream files before patching, especially:

```text
obra/superpowers
  skills/subagent-driven-development/SKILL.md
  skills/subagent-driven-development/implementer-prompt.md
  skills/using-superpowers/references/codex-tools.md
  skills/writing-skills/SKILL.md

NandhaKishorM/laya
  README.md
  laya/router.py
  pyproject.toml
```

Do not assume line numbers or APIs remain stable. Prefer semantic discovery and tests.

---

# 60. Completion instruction to the coding agent

Implement this incrementally and keep the fork usable after every phase.

Before declaring the feature complete, demonstrate with automated tests and a real routed SDD run that:

1. a trivial coding task can execute on a configured cheap/local/free tier,
2. the subagent is explicitly dispatched on that model rather than inheriting the controller,
3. a failed verified attempt escalates with evidence,
4. an infrastructure failure stays within the same tier where possible,
5. an architecture/final-review task respects its minimum capability floor,
6. telemetry identifies the cheapest successful tier and total attempts.

The desired end state is not “always use the cheapest model.”

The desired end state is:

> **Use the cheapest model that can demonstrably complete the work to Superpowers' existing quality standard, and escalate only when evidence justifies it.**
