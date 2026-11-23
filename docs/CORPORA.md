# Ethics conditioning corpora

Built 2026-11-16 (decisions D-49 to D-55). The texts in `corpora/<name>/*.md` are gitignored. Only the
`manifest.yaml` files are tracked. Every manifest source carries `tier: sep | primary`.

## Sources and sizes

Token counts use the repo counter `approx_tokens` (ceil(chars/4)), the same counter that budgets use.

| Framework | Sources | SEP tier (tokens) | Primary tier (tokens) | Total |
|---|---|---|---|---|
| deontology | 4 | 2 (44.5k) | 2 (129.2k) | 173.8k |
| act_utilitarian | 8 | 5 (69.5k) | 3 (97.1k) | 166.6k |
| rule_utilitarian | 2 | 1 (20.5k) | 1 (19.2k) | 39.7k |
| virtue | 4 | 3 (62.2k) | 1 (116.5k) | 178.6k |
| divine_command | 3 | 2 (48.9k) | 1 (9.1k) | 58.0k |
| care | 1 | 1 (20.6k) | 0 | 20.6k |
| contractualism | 4 | 2 (33.5k) | 2 (37.9k) | 71.4k |
| *placebo (reference)* | 236 | — | — | 202.2k |
| *_candidates/natural_law* | 2 | 2 (40.2k) | — | 40.2k |
| *_candidates/buddhist* | 1 | 1 (17.6k) | — | 17.6k |

### SEP tier (entry slug, sections kept)

The revision year in the manifests is the "substantive revision" year shown on the entry.

- **deontology**: `ethics-deontological` (2024), `kant-moral` (2025, §16 "AI Ethics" dropped)
- **act_utilitarian**: `consequentialism` (2023), `utilitarianism-history` (2025), `bentham` §2–5 (2026),
  `mill-moral-political` §2 "Mill's Utilitarianism" (2026), `sidgwick` §2 "The Methods of Ethics" (2023)
- **rule_utilitarian**: `consequentialism-rule` (2023)
- **virtue**: `ethics-virtue` (2026), `aristotle-ethics` (2026), `moral-character` (2019)
- **divine_command**: `voluntarism-theological` (2025), `religion-morality` (2024)
- **care**: `feminism-ethics` (2025). SEP has no dedicated care-ethics entry. Care is §2.2 of this entry,
  and the rest is feminist ethics more broadly.
- **contractualism**: `contractualism` (2018), `contractarianism` (2025)

What the cleaner keeps and drops: it keeps the preamble and main text, with headings as Markdown. It drops
navigation, the TOC, the bibliography, Academic Tools, Other Internet Resources, Related Entries, footnote
markers, and the separate notes and supplement pages. Inline author–year citations stay.

### Primary tier (all cited by the corpus's own SEP entries)

| Framework | Work | Edition / source | Extent |
|---|---|---|---|
| deontology | Kant, *Groundwork* (1785) | Abbott tr., Gutenberg #5682 | whole |
| deontology | Kant, *Critique of Practical Reason* (1788) | Abbott tr., Gutenberg #5683 | whole |
| act_utilitarian | Mill, *Utilitarianism* (1861) | 7th ed., Gutenberg #11224 | whole, footnotes dropped |
| act_utilitarian | Bentham, *IPML* (1789) | Wikisource | ch. I–V |
| act_utilitarian | Sidgwick, *Methods of Ethics* (1907) | 7th ed., Gutenberg #46743 | Book IV only |
| rule_utilitarian | Berkeley, *Passive Obedience* (1712) | Fraser ed. 1901 vol. IV, Internet Archive OCR | To the Reader + Discourse |
| virtue | Aristotle, *Nicomachean Ethics* | Ross tr. 1925, Wikisource | Books I–X |
| divine_command | Plato, *Euthyphro* | Jowett tr., Gutenberg #1642 | dialogue only |
| contractualism | Rousseau, *Social Contract* (1762) | Cole tr. 1913, Gutenberg #46333 | Books I–II |
| contractualism | Hobbes, *Leviathan* (1651) | Gutenberg #3207 | ch. XIII–XV |

Gutenberg headers and licence text, translators' and editors' introductions, footnotes and hard wraps are
stripped. Gutenberg's main site was down (503), so files came from the PGLAF mirror `gutenberg.pglaf.org`.

## Licences

- **SEP** ([Terms of Use](https://plato.stanford.edu/info.html)): each user gets a "royalty-free
  non-exclusive limited license to read, download, make copies, print, search, or link to the full text of
  each SEP entry", and may "electronically distribute an SEP entry to computers and mobile devices that
  are owned and controlled by the User". Wider electronic distribution is limited to private
  correspondence or closed groups of at most 30 people, and it must be non-commercial. Local copies for
  personal research are therefore within the terms. **Do not commit or publish the texts.** The terms do
  not address model training. Treat any LoRA adapter trained on SEP text as derived material and do not
  publish it without checking.
- **Primary texts**: public domain in the USA. The Ross (d. 1971) and Cole (d. 1959) translations are
  *not* PD in life+70 jurisdictions; this is noted in their manifests.

## Gaps and asymmetries (findings, not fixed)

- **care has no primary tier.** The founding texts (Gilligan, Noddings, Held, Tronto) are all in
  copyright. care is also the smallest corpus (21k), below the template's ~50k LoRA floor.
- **rule_utilitarian is thin** (40k). There is no PD rule-consequentialist classic: Brandt, Harsanyi and
  Hooker are all in copyright, and Austin's *Province* is not transcribed on Wikisource. Berkeley's
  *Passive Obedience* is the only PD text the SEP entry cites. It is theological and OCR-noisy (split
  hyphenations, stray page artefacts). Mill is cited too but sits only in act_utilitarian, so the two
  utilitarian arms are not padded with the same text.
- **contractualism primary is contractarian/social-contract, not Scanlonian.** No PD source exists for
  "what we owe to each other". Hobbes is closer to the contractarianism entry than to the registered
  summary.
- **divine_command primary is the Euthyphro**, which is the classic *objection* to divine command theory.
  No PD English voluntarist primary (Ockham, Scotus) was available. `religion-morality` also covers
  natural law, Kant and Nietzsche, so it overlaps the natural_law candidate.
- **Register differs across tiers**: SEP is modern analytic prose, while primaries are 17th–19th-century
  prose or translations. Frameworks whose corpora are mostly primary (deontology 74%, virtue 65%) will
  condition differently from SEP-only ones (care, and the candidates).
- The deontology SEP entry opens with a section on consequentialism ("Deontology's Foil"). Entries were
  kept whole apart from the cuts listed above.

## What retrieval actually uses (2026-11-22)

For corpus review. `uv run python scripts/retrieval_review.py` replays BM25 retrieval for the 177
harmful prompts of the v0 runs (k and budget from `abl-rag-frameworks.yaml`) and counts which source
each excerpt came from. It also writes the most-retrieved chunks, with their text, to
`labels/corpus-review-top-chunks.md` (gitignored, since it quotes corpus text).

Reading the tables: in the large corpora a third or more of the chunks are ever retrieved, and SEP
entries are over-represented among the excerpts relative to their size (virtue: SEP is 34% of chunks
and 53% of excerpts; deontology: 25% and 47%). Blocks are filled to the budget
with next-best chunks (D-46), and the eight most-retrieved chunks per framework are mostly general passages, not
passages about harm (check the local file). These are descriptive counts. Nothing here has been used to change a corpus.

**virtue**: 1133 chunks, 432 ever retrieved (38%); the 10 most-retrieved chunks fill 18% of all excerpt slots.

| Source | Tier | Share of corpus chunks | Share of retrieved excerpts |
|---|---|---|---|
| Nicomachean Ethics, tr. W. D. Ross (1925) | primary | 66% | 47% |
| Moral Character | sep | 13% | 21% |
| Aristotle’s Ethics | sep | 13% | 17% |
| Virtue Ethics | sep | 8% | 15% |

**deontology**: 1125 chunks, 400 ever retrieved (36%); the 10 most-retrieved chunks fill 14% of all excerpt slots.

| Source | Tier | Share of corpus chunks | Share of retrieved excerpts |
|---|---|---|---|
| The Critique of Practical Reason, tr. T. K. Abbott | primary | 52% | 34% |
| Kant’s Moral Philosophy | sep | 15% | 24% |
| Deontological Ethics | sep | 10% | 23% |
| Fundamental Principles of the Metaphysic of Morals (Groundwork), tr. T. K. Abbott | primary | 24% | 19% |

**care**: 128 chunks, 122 ever retrieved (95%); the 10 most-retrieved chunks fill 38% of all excerpt slots.

| Source | Tier | Share of corpus chunks | Share of retrieved excerpts |
|---|---|---|---|
| Feminist Ethics | sep | 100% | 100% |

**contractualism**: 457 chunks, 306 ever retrieved (67%); the 10 most-retrieved chunks fill 20% of all excerpt slots.

| Source | Tier | Share of corpus chunks | Share of retrieved excerpts |
|---|---|---|---|
| The Social Contract, Books I-II, tr. G. D. H. Cole (1913) | primary | 30% | 40% |
| Contractualism | sep | 32% | 38% |
| Leviathan, chapters XIII-XV | primary | 23% | 11% |
| Contractarianism | sep | 14% | 10% |

**divine_command**: 362 chunks, 253 ever retrieved (70%); the 10 most-retrieved chunks fill 19% of all excerpt slots.

| Source | Tier | Share of corpus chunks | Share of retrieved excerpts |
|---|---|---|---|
| Religion and Morality in Western Philosophy | sep | 44% | 47% |
| Euthyphro, tr. B. Jowett | primary | 16% | 29% |
| Theological Voluntarism | sep | 40% | 24% |

**rule_utilitarian**: 252 chunks, 204 ever retrieved (81%); the 10 most-retrieved chunks fill 23% of all excerpt slots.

| Source | Tier | Share of corpus chunks | Share of retrieved excerpts |
|---|---|---|---|
| Rule Consequentialism | sep | 50% | 64% |
| Passive Obedience (1712), in Works vol. IV, ed. A. C. Fraser (1901); OCR text | primary | 50% | 36% |

**act_utilitarian**: 1069 chunks, 425 ever retrieved (40%); the 10 most-retrieved chunks fill 15% of all excerpt slots.

| Source | Tier | Share of corpus chunks | Share of retrieved excerpts |
|---|---|---|---|
| The Methods of Ethics (7th ed.), Book IV: Utilitarianism | primary | 28% | 23% |
| Consequentialism | sep | 10% | 21% |
| Utilitarianism (7th ed., 1879) | primary | 23% | 18% |
| The History of Utilitarianism | sep | 9% | 10% |
| Henry Sidgwick (sections 2 only) | sep | 8% | 9% |
| Mill’s Moral and Political Philosophy (sections 2 only) | sep | 9% | 9% |
| Jeremy Bentham (sections 2, 3, 4, 5 only) | sep | 5% | 6% |
| An Introduction to the Principles of Morals and Legislation, chapters I-V | primary | 7% | 4% |

## Proposed balancing rule (not implemented)

Equalise within each tier rather than in total. For each framework:

- SEP tier: cap at the smallest framework's SEP total, sampling whole sections round-robin across entries.
- Primary tier: cap at a fixed value P, for example 40k, and record "0" where none exists.
- For LoRA, train on `min(total, T)` tokens with a common T (for example 40k ≈ rule_util's size) rather
  than cycling small corpora up to `target_tokens`. Alternatively, run two LoRA arms, SEP-only (balanced
  ~20k) and SEP+primary, so the primary-tier asymmetry becomes a measured factor.

Prompt and RAG are already budget-equalised per call; RAG pads to the budget (D-46). Imbalance there
changes only how varied the retrieved pool is.

## Theology candidates (`corpora/_candidates/`, not registered)

- `natural_law`: `natural-law-ethics` (2025) and `aquinas-moral-political` (2021).
- `buddhist`: `ethics-indian-buddhism` (2024, §8 "AI and Technology" dropped).
- **Islamic and Jewish ethics: not built.** SEP has no ethics entry for either. `arabic-islamic-religion`
  is philosophy of religion, and `maimonides` has one "Practical Philosophy" section.
- Their summaries are unreviewed placeholders in the manifests. Check them with
  `uv run ethica corpus check natural_law --root corpora/_candidates`.
- `.gitignore` currently ignores `_candidates/*/manifest.yaml`.

## Rebuild

The fetch and clean code lives outside this repo in `~/Developer/ethica-private/` (`fetch.py`, `clean.py`,
`build.py`; raw responses cached under `raw/`). From that directory, `uv run python build.py [name ...]`
rewrites the `.md` files and `manifest.yaml` here. Then run `uv run ethica corpus check <name>`.
