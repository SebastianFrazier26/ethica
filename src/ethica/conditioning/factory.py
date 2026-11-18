"""Variant -> Conditioner."""

from __future__ import annotations

from pathlib import Path

from ethica.conditioning.bm25 import BM25Index
from ethica.conditioning.corpus import framework_from_manifest, load_corpus
from ethica.conditioning.lora import LoRAConditioner
from ethica.conditioning.prompt import NoneConditioner, PromptConditioner
from ethica.conditioning.rag import RAGConditioner, build_index
from ethica.types import Conditioner, Variant


def make_conditioner(
    variant: Variant, corpora_root: str | Path, budget_tokens: int, k: int
) -> Conditioner:
    fw = variant.framework
    if fw.kind == "none" or variant.method == "none":
        return NoneConditioner()

    corpus = load_corpus(Path(corpora_root) / fw.name)
    # the manifest on disk is authoritative for summary/display_name
    framework = framework_from_manifest(corpus.manifest, corpus.root)

    match variant.method:
        case "prompt":
            return PromptConditioner(framework, corpus, budget_tokens)
        case "rag":
            idx_path = variant.rag_index_path
            index = (
                BM25Index.load(idx_path)
                if idx_path and Path(idx_path).is_file()
                else build_index(corpus)
            )
            return RAGConditioner(framework, index, k, budget_tokens)
        case "lora" | "lora+prompt":
            if not variant.adapter_path:
                raise ValueError(
                    f"{variant.variant_id}: method {variant.method} needs adapter_path"
                )
            inner = (
                PromptConditioner(framework, corpus, budget_tokens)
                if variant.method == "lora+prompt"
                else None
            )
            return LoRAConditioner(variant.adapter_path, inner)
    raise ValueError(f"unknown method {variant.method!r}")
