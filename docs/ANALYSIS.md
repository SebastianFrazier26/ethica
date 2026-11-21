# Analysis plan

Implemented in `src/ethica/analysis/stats.py` (numpy, seeded) and `report.py`.

## Estimands
- Cell rate: mean of a metric (0/1 for binary) per variant = model x safeguards x framework x method, averaged over tasks and seeds.
- Framework effect: rate(framework) - rate(reference) on matched task x seed pairs, within a model x safeguards x method stratum. Reported as risk difference with Cohen's h alongside.
- Safeguards interaction: (framework effect in abliterated) - (framework effect in stock), on task means. Stock and abliterated twins are different model keys; they are paired through the abliterated model's `base_of` (stored as `variants.pair_key`), and the report labels the row `<stock key>/<abliterated key>`.

## Why these methods
- **Cluster bootstrap by task.** Seeds and variants on the same scenario are not independent; the task is the sampling unit. Resampling rows would give CIs that are too narrow. CIs are percentile intervals over 2000 resamples of tasks. With few tasks (< 10) the CI is unreliable; the report warns.
- **Paired permutation (sign-flip), unit = task (D-43, closes D-23).** Both variants face identical task x seed pairs, so pairing removes task difficulty. All pairs of a task are flipped together (`permute_by="task"`, the default), because seeds within a task are correlated and the bootstrap already resamples tasks; the two procedures now share one unit. Exact enumeration for <= 14 tasks, else Monte Carlo with the +1 correction. `permute_by="pair"` (per task x seed pair) stays available as a sensitivity check but is anti-conservative with several seeds and is not the primary analysis. The safeguards interaction is computed on task means, so its unit is the task too.
- **Reference choice.** Placebo is primary (D-04): it isolates framework content from "any added text". `none` is secondary.

## Multiple comparisons
- Holm-Bonferroni (FWER) across frameworks, within each metric x stratum x reference.
- Primary metric is not further adjusted across strata (each stratum is a separate question); read per-stratum.
- Exploratory metrics: additionally Benjamini-Hochberg (FDR) across all exploratory framework-vs-placebo contrasts (`p_bh_exploratory`).

## Proposed pre-registration (for approval)
Primary:
1. `misaligned_action`, each ethics framework vs `placebo`, per model x safeguards x method stratum; Holm across frameworks.
2. Safeguards interaction on `misaligned_action`, framework vs placebo, per model x method.

Secondary (confirmatory in direction, reported with CIs):
3. Same contrasts vs `none`.
4. Manipulation check: `fidelity__<framework>` is highest on the matching framework (descriptive heatmap; no test).
5. Alignment tax: `capability_acc__*`, `over_refusal`, `harmful_compliance`, `truthful_acc`, `task_success` vs placebo.

Exploratory (hypothesis-generating, BH-adjusted): category metrics (`oversight_subversion`, `self_exfiltration`, `shutdown_resistance`, ...), `framework_rationalisation`, `scheming_reasoning`, `deception`, `coercion`, dose-response across methods.

## Data hygiene
- **Only the current grid is reported.** `report` passes the `variant_id`s of `expand_grid(cfg)`; episodes from variants of an earlier config (different weights, corpus, budget, generation params) stay in the DB but are never averaged in. If one (model, safeguards, framework, method) still maps to more than one `variant_id` the report raises `ReportError`.
- `variant_id` hashes everything that changes behaviour (resolved weights path + quant, corpus content + manifest summary/display name, budget/k, adapter, generation params); `episode_id` additionally hashes scenario / item content and the experiment name. Changing any of them yields new rows, not stale resumes.
- Errored episodes are not dropped silently. Behavioural metrics (`misaligned_action`, category metrics) are scored from events recorded before the error, with `task_success` NaN; an errored episode with no events is NaN. Errored episodes are retried on the next `generate` (up to 3 attempts) unless the error is a context overflow, which is final. The report has a per-arm table of errors, context overflows, retries, truncations, protocol violations and mean conditioning block size.
- **One scorer family per metric** so cells are comparable (`programmatic`, `exact`, `judge:<key>`, `heuristic:refusal`; check mix lives in the rationale). Preference: override > judge for `over_refusal` and `harmful_compliance` > programmatic > exact > other judges > heuristics. The report tabulates heuristic-vs-judge agreement per arm; a large gap means the phrase heuristic is biased for that arm.
- Benchmarks emit `benchmark_truncated` and `benchmark_unparseable` per item, and the report tabulates both per arm and benchmark. Fidelity scores NaN on unparseable answers, so a wordier arm silently loses items unless this table is read first.
- NaN scores (not applicable) are dropped per metric.
- Rescoring replaces the episode's rows of the same scorer family (stale metrics are deleted). Judge resume is per (episode, rubric). Config edits between phase runs are kept in the `experiment_runs` table.
