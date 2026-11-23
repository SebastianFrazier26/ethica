"""Corpus-review aid: which sources and chunks fill the RAG excerpts on the harmful prompts.

BM25 retrieval is deterministic, so this replays it with the harness's own code (as
scripts/lexical_analysis.py does) for each ethics corpus.

    uv run python scripts/retrieval_review.py

Prints a per-source Markdown table (titles and shares only; pasted into docs/CORPORA.md) and
writes the most-retrieved chunks, with text, to labels/corpus-review-top-chunks.md. That file
quotes corpus text, so it lives under the gitignored labels/ and must not be published.
"""

from __future__ import annotations

from collections import Counter

import lexical_analysis as la

from ethica.benchmarks import registry as bench_registry
from ethica.conditioning.corpus import load_corpus
from ethica.conditioning.rag import RAGConditioner, build_index
from ethica.config import load_config
from ethica.types import Message

TOP_CHUNKS = 8
OUT = la.ROOT / "labels" / "corpus-review-top-chunks.md"


def main() -> None:
    cfg = load_config(la.CONFIGS[0])
    prompts, _ = la.harmful_tasks()
    local = [
        f"# Most-retrieved chunks per framework ({len(prompts)} harmful prompts, "
        f"k={cfg.rag_k}, budget={cfg.budget_tokens}). LOCAL ONLY: quotes corpus text."
    ]
    for fw in la.ETHICS:
        corpus = load_corpus(la.ROOT / "corpora" / fw)
        index = build_index(corpus)
        cond = RAGConditioner(corpus.framework, index, cfg.rag_k, cfg.budget_tokens)
        text = dict(zip(index.ids, index.texts, strict=True))
        source = {d.source.file: d.source for d in corpus.documents}
        chunks_per_file = Counter(i.split("#")[0] for i in index.ids)
        file_slots: Counter[str] = Counter()
        chunk_slots: Counter[str] = Counter()
        for bench, prompt in prompts.values():
            system = bench_registry.get(bench).system_prompt
            msgs = ([Message("system", system)] if system else []) + [Message("user", prompt)]
            cond.retrieve(msgs)
            for hit in cond.last_hits:
                file_slots[hit.split("#")[0]] += 1
                chunk_slots[hit] += 1
        total, n_chunks = sum(file_slots.values()), len(index.ids)
        top10 = sum(c for _, c in chunk_slots.most_common(10)) / total
        print(
            f"\n**{fw}**: {n_chunks} chunks, {len(chunk_slots)} ever retrieved "
            f"({len(chunk_slots) / n_chunks:.0%}); the 10 most-retrieved chunks fill "
            f"{top10:.0%} of all excerpt slots.\n\n"
            "| Source | Tier | Share of corpus chunks | Share of retrieved excerpts |\n"
            "|---|---|---|---|"
        )
        for file, n in sorted(chunks_per_file.items(), key=lambda kv: -file_slots[kv[0]]):
            s = source[file]
            share = file_slots[file] / total
            print(f"| {s.title} | {s.tier or '-'} | {n / n_chunks:.0%} | {share:.0%} |")
        local.append(f"\n## {fw}\n")
        for cid, c in chunk_slots.most_common(TOP_CHUNKS):
            excerpt = " ".join(text[cid].split())[:420]
            title = source[cid.split("#")[0]].title
            local.append(f"- **{c}/{len(prompts)} prompts** · {title} · `{cid}`\n  > {excerpt}…")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(local) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(la.ROOT)}")


if __name__ == "__main__":
    main()
