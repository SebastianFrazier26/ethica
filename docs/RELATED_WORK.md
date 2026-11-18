# Related work, novelty, literature shortlist, threats to validity

Researched 2026-11-16 by a research agent using web search; written up from its report.
Items marked **[U]** were not verified. Public-domain (PD) claims are **unverified**; the
agent made them from memory. Check each one before any text enters a corpus. Findings may be stale.

## 1. Novelty

No study was found that conditions small open-weight models on **one ethical framework's literature**
(prompt / RAG / LoRA) and measures **agentic misalignment** against a no-conditioning control and a
length-matched placebo, with stock vs abliterated variants. The search was only a handful of queries, so
re-check before claiming novelty.

**Closest work, read first:** Coleman et al., *Six Llamas: Comparative Religious Ethics Through
LoRA-Adapted Language Models*, arXiv 2604.18404 (Apr 2026). Five LoRA variants of Llama-3.1-8B, each
trained on one religion's texts, plus a control. It measures ethical-reasoning answers on 17 prompts.
It has no safety or agentic outcomes and no placebo.

Other adjacent work (verified):
| Work | Finding | How ethica differs |
|---|---|---|
| *Pull or Not to Pull?* arXiv 2508.07284 | 10 moral-philosophy "expert" system prompts on trolley dilemmas across 14 LLMs | no safety or agentic outcomes |
| Structured Moral Reasoning, 2506.14948 | reasoning scaffolds improve moral accuracy | accuracy, not misalignment |
| arXiv 2411.11731 | framework-persuasion probe on dilemmas | static only |
| MoralReason, 2511.12271 | RL toward utilitarian or deontological reasoning; its base model leans virtue ethics | **the control is not framework-neutral, so measure the baseline lean** |
| Plaza-del-Arco, Röttger, Scherrer, 2509.08075 | persona prompts shift false refusal | relevant to the over-refusal tax |
| Emergent Misalignment, Betley et al. 2502.17424 | narrow fine-tuning causes broad misalignment | **key LoRA confound** |
| Persona Vectors, 2507.21509; Open Character Training, 2511.01689 | character can be steered by activation directions or training | possible follow-up analysis |
| Constitutional AI, 2212.08073; Deliberative Alignment, 2412.16339 | principles trained in | ours are whole normative frameworks, not a safety spec |
| Delphi 2110.07574; MoralChoice 2307.14324; ETHICS 2008.02275 | moral judgement benchmarks | used as static benchmarks |

[U]: MoralBench; persona-jailbreak figures attributed to Zhang et al.

## 2. Eval literature behind the scenarios (verified)

- Apollo in-context scheming (2412.04984): frontier models only, mostly under strong goal nudges.
- Anthropic agentic misalignment (2510.05179): 16 models in a goal × urgency factor grid. Models misbehaved less when they stated they were being tested.
- Alignment faking (2412.14093). Koorndijk (2506.21584) reports alignment faking in Llama-3.1-8B; single author, not peer-reviewed. Greenblatt et al. saw no compliance gap in small Llama models.
- Sandbagging (2406.07358). Shutdown resistance, Palisade (2509.14260): results depend heavily on prompt wording.
- Sycophancy, Sharma et al. (2310.13548). MACHIAVELLI (2304.03279).
- Refusal direction / abliteration, Arditi et al. (2406.11717). Abliteration capability comparison (2512.13655): GSM8K is the most sensitive capability measure, and short generation budgets can inflate GSM8K scores for abliterated models.
- Reward hacking: EvilGenie (2511.21654), SpecBench (2605.21384), Reward Hacking Benchmark (2605.02964). Lesson: hold out tests the model can't see, and treat passing the visible tests while failing the held-out ones as a programmatic hack signal.
- Eval awareness, Needham (2505.23836).

**Small-model capability floor:** no published Apollo-style scheming rates were found for models of 8B or smaller.
Expect near-zero base rates. Design for it: nudged scenario variants, many seeds, paraphrases.

## 3. Existing harnesses

- **UK AISI Inspect** (`inspect_ai` + `inspect_evals`, MIT per GitHub metadata): borrow its task/solver/scorer
  split and log format. Adopting it as the runner instead of our own loop is **your call** (logged as an open
  question in DECISIONS). v0 builds its own runner, per your build-from-scratch norm.
- Anthropic's public agentic-misalignment repo: borrow its factor-grid idea.
- lm-evaluation-harness, OpenAI evals, HELM: not researched [U].

## 4. Literature shortlist per framework (for tomorrow)

PD status is from the agent's memory and **must be verified**.

- **Deontology:** Kant, *Groundwork* and *Critique of Practical Reason* (Abbott trans., PD); Prichard 1912. Ross, *The Right and the Good* (1930) is borderline.
- **Act utilitarianism:** Bentham, *Principles*; Mill, *Utilitarianism*; Sidgwick, *Methods of Ethics* (7th ed.); Moore, *Principia Ethica*.
- **Rule utilitarianism:** PD material is thin: Mill ch. 5 and *On Liberty*. Hooker, Brandt, Harsanyi and Hare are in copyright. ⚠ Risk of near-identical corpora with act-utilitarianism.
- **Virtue ethics:** Aristotle, *Nicomachean Ethics* (Ross trans.; verify edition); Plato (Jowett); Cicero, *De Officiis*; Epictetus; Marcus Aurelius (Long trans.); Hume.
- **Theological:** split it. Thomist natural law: *Summa* I-II (Dominican Fathers trans., PD). Divine command: KJV selections, Kierkegaard. Optional non-Christian traditions: Maimonides (Friedländer trans.), Confucian texts (Legge trans.). *Six Llamas* already covers five religions.
- **Care ethics:** core texts (Gilligan, Noddings, Held, Tronto) are in copyright. No open-access substitute verified.
- **Contractualism:** Rawls and Scanlon are in copyright. PD precursors are Hobbes, Locke and Rousseau, but Hobbes is a poor stand-in for Scanlon.
- **SEP:** authors keep copyright and license Stanford exclusively (from archived policy pages). Read it, but **don't use it as a training or RAG corpus**.
- **Keeping corpora comparable:** fix the token budget using the target tokenizer, match the ratio of primary to secondary texts, measure register per arm, apply identical chunking and dedup, and use **≥2 placebos**: register-matched non-ethical PD prose, plus off-framework ethics.

## 5. Threats to validity (ranked)

1. **Capability floor and noise.** Rates may be near zero in every arm. Mitigation: nudged variants, many seeds and paraphrases, CIs, a pre-registered minimum effect size, report nulls.
2. **Placebo and surface confounds** (length, register, format, the act of LoRA training itself). Mitigation: multiple placebos, token matching, register metrics.
3. **Scenario leakage and judge error.** Mitigation: programmatic ground truth, human spot-checks, paraphrase sets.
4. **LoRA side effects** (Emergent Misalignment).
5. **Corpus overlap and mislabelled frameworks** (act vs rule utilitarianism).
6. **Stock models are not framework-neutral.** Measure the baseline lean with the fidelity probe on `none`.
7. **Abliteration confounds**: tool-use variance, KL drift, generation length.
8. **Eval awareness and scratchpad format effects.**
9. **Fidelity-probe validity.** Test behaviour where frameworks diverge, not vocabulary.
10. **Multiple comparisons**: 7 frameworks × 3 methods × 2 safeguard levels × ~8 behaviours.
11. **Generalisation** from small models to large ones.
12. **Licence risk** from in-copyright corpora.
