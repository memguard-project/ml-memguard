# OOM Prediction Model — Accuracy Reference

Accuracy of the `oom_classifier_v1` GBT model on the held-out test split
(2,000 sessions; 80/20 train-test from 10,000 synthetic sessions, seed 42).

Model architecture: `GradientBoostingClassifier(n_estimators=100, max_depth=4,
learning_rate=0.1, subsample=0.8)` + `CalibratedClassifierCV(method="isotonic", cv=3)`.

---

## Overall metrics

| Version | Train size | Test size | AUC-ROC | Precision@0.70 | Recall@0.92 | Date |
|---------|------------|-----------|---------|----------------|-------------|------|
| v1      | 8,000      | 2,000     | 0.9942  | 0.9980         | 0.8325      | 2026-04-12 |

**Quality gates** (CI enforced): AUC-ROC ≥ 0.85 / Precision@0.70 ≥ 0.75 / Recall@0.92 ≥ 0.80

---

## Per-scenario breakdown

The training data is generated from three synthetic failure scenarios (50/25/25 mix).
Below are test-split metrics evaluated per scenario independently.

| Scenario           | Test n | OOM rate | AUC-ROC | Precision@0.70 | Recall@0.92 | Notes |
|--------------------|-------:|---------:|---------|----------------|-------------|-------|
| `gradual_fill`     | 1,000  | 9.1%     | 1.0000  | 1.0000         | 1.0000      | Slow monotonic KV growth — perfectly separable; high utilization at prediction point |
| `burst_long_seqs`  | 499    | 58.3%    | 1.0000  | 1.0000         | 1.0000      | High-velocity 32k-token bursts — velocity + eviction signals are unambiguous |
| `fragmentation_trap` | 501  | 41.7%    | 0.9210  | 0.9912         | 0.5263      | OOM fires at 65% fill; low utilization at prediction point makes 0.92 threshold hard to reach |

### Reading the fragmentation_trap result

`fragmentation_trap` models the SGLang/vLLM failure mode where high fragmentation
causes OOM at moderate utilization (30–60%).  Because utilization is low when the
OOM is predicted, the model's probability output is also lower — most fragmentation-trap
OOM sessions score between 0.70 and 0.91, not above 0.92.

**Practical implication**: for RadixAttention or high-fragmentation workloads, rely on
the `shed_load` threshold (0.70, Precision 99.1%) rather than the `restart` threshold
(0.92).  Tighten `shed_load_threshold` to 0.65 for reasoning model deployments:

```python
safe = guard_sglang(runtime)
safe.monitor.shed_load_threshold = 0.65   # catch fragmentation-trap OOMs earlier
safe.monitor.on_shed_load = lambda u: lb.reduce_weight(replica_id, 0)
```

---

## Feature importance

Derived from GBT feature importance scores on the v1 model (normalized, descending):

| Feature               | Importance | Role |
|-----------------------|------------|------|
| `utilization`         | 0.41       | Derived: kvcache_mb / (weights + kv + activations + ctx) |
| `kv_velocity_mbps`    | 0.27       | KV cache growth rate — primary leading indicator |
| `eviction_rate`       | 0.14       | Preemptions/sec — elevated before OOM in all scenarios |
| `fragmentation_ratio` | 0.10       | Dominant signal for fragmentation_trap scenario |
| `near_miss_count`     | 0.04       | Allocation successes with < 512 MB headroom |
| `avg_seq_len`         | 0.03       | Proxy for burst_long_seqs scenario |
| `preemption_count`    | 0.01       | Cumulative preemptions; correlates with eviction_rate |

---

## Training data scenarios

| Scenario             | Mix weight | OOM rate | Key signal | Representative failure |
|----------------------|------------|----------|------------|------------------------|
| `gradual_fill`       | 50%        | ~10%     | utilization approaching 1.0 | Llama-70B at max batch size, context filling over hours |
| `burst_long_seqs`    | 25%        | ~60%     | kv_velocity_mbps spike | Sudden arrival of 32k-token coding requests |
| `fragmentation_trap` | 25%        | ~40%     | fragmentation_ratio > 0.30 | SGLang RadixAttention dead branches from reasoning models |

---

## Version history

| Version | Train size | AUC-ROC | Precision@0.70 | Recall@0.92 | Change |
|---------|------------|---------|----------------|-------------|--------|
| v1      | 8,000      | 0.9942  | 0.9980         | 0.8325      | Initial model |
