"""Persistence layer. All access goes through the ORM/Core with bound parameters."""

from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Collection, Iterator, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, delete, event, func, inspect, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ethica.tracking.models import (
    Base,
    EnvEventRow,
    Episode,
    Experiment,
    ExperimentRun,
    MessageRow,
    ScoreRow,
)
from ethica.tracking.models import Variant as VariantRow
from ethica.types import EnvEvent, EpisodeResult, Message, Score, ToolCall, Variant

DEFAULT_DB_URL = "sqlite:///runs/ethica.db"


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def scorer_family(scorer: str) -> str:
    """`programmatic:<checks>` -> `programmatic`; `judge:<key>` and `heuristic:<kind>` are kept
    whole, since different judges / heuristics are different instruments."""
    if scorer.startswith(("judge:", "heuristic:")):
        return scorer
    return scorer.split(":", 1)[0]


class Store:
    def __init__(self, db_url: str | None = None) -> None:
        url = make_url(db_url or os.environ.get("ETHICA_DB_URL") or DEFAULT_DB_URL)
        kwargs: dict[str, Any] = {}
        if url.get_backend_name() == "sqlite":
            if url.database in (None, "", ":memory:"):
                # in-memory DBs are per-connection; share one so tests see a single DB
                kwargs.update(poolclass=StaticPool, connect_args={"check_same_thread": False})
            else:
                Path(url.database).parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = create_engine(url, **kwargs)
        if url.get_backend_name() == "sqlite":

            @event.listens_for(self.engine, "connect")
            def _pragmas(dbapi_conn: Any, _rec: Any) -> None:
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA foreign_keys=ON")
                cur.execute("PRAGMA journal_mode=WAL")
                cur.close()

        Base.metadata.create_all(self.engine)
        self._add_missing_columns()
        self._session = sessionmaker(self.engine, expire_on_commit=False)

    def _add_missing_columns(self) -> None:
        # create_all never alters existing tables; this one additive column is the only
        # schema drift worth migrating in place (older DBs fall back to model_key pairing).
        cols = {c["name"] for c in inspect(self.engine).get_columns("variants")}
        if "pair_key" not in cols:
            with self.engine.begin() as conn:
                conn.execute(text("ALTER TABLE variants ADD COLUMN pair_key VARCHAR"))

    def close(self) -> None:
        self.engine.dispose()

    # ---- experiments / variants ----

    def create_experiment(
        self,
        name: str,
        config: dict[str, Any] | None = None,
        git_rev: str | None = None,
        experiment_id: str | None = None,
    ) -> str:
        eid = experiment_id or uuid.uuid4().hex[:12]
        with self._session.begin() as s:
            if s.get(Experiment, eid) is None:
                s.add(
                    Experiment(id=eid, name=name, config_json=_dumps(config or {}), git_rev=git_rev)
                )
        return eid

    def record_run(self, experiment_id: str, phase: str, config: dict[str, Any]) -> None:
        with self._session.begin() as s:
            s.add(
                ExperimentRun(experiment_id=experiment_id, phase=phase, config_json=_dumps(config))
            )

    def runs(self, experiment_id: str) -> list[ExperimentRun]:
        with self._session() as s:
            return list(
                s.scalars(
                    select(ExperimentRun)
                    .where(ExperimentRun.experiment_id == experiment_id)
                    .order_by(ExperimentRun.id)
                )
            )

    def upsert_variant(self, variant: Variant) -> None:
        """Insert-only: variant_ids are content-addressed, so an existing row already describes
        this variant and must not have its provenance rewritten."""
        m, f = variant.model, variant.framework
        fields: dict[str, Any] = {
            "model_key": m.key,
            "hf_repo": m.hf_repo,
            "safeguards": m.safeguards,
            "family": m.family,
            "pair_key": m.base_of or m.key,
            "params_b": m.params_b,
            "framework": f.name,
            "framework_kind": f.kind,
            "method": variant.method,
            "adapter_path": variant.adapter_path,
            "spec_json": _dumps(asdict(variant)),
        }
        with self._session.begin() as s:
            row = s.get(VariantRow, variant.variant_id)
            if row is None:
                s.add(VariantRow(variant_id=variant.variant_id, **fields))

    # ---- episodes ----

    def has_episode(self, episode_id: str) -> bool:
        with self._session() as s:
            return s.get(Episode, episode_id) is not None

    def completed_episode_ids(self, experiment_id: str) -> set[str]:
        with self._session() as s:
            rows = s.scalars(
                select(Episode.episode_id).where(Episode.experiment_id == experiment_id)
            )
            return set(rows)

    def episode_status(self, experiment_id: str) -> dict[str, tuple[str | None, int]]:
        """episode_id -> (error, attempts) for resume decisions."""
        with self._session() as s:
            rows = s.execute(
                select(Episode.episode_id, Episode.error, Episode.meta_json).where(
                    Episode.experiment_id == experiment_id
                )
            )
            return {eid: (err, int(json.loads(meta).get("attempts", 1))) for eid, err, meta in rows}

    def delete_episode(self, episode_id: str) -> None:
        """Messages, events and scores go with it via ON DELETE CASCADE."""
        with self._session.begin() as s:
            s.execute(delete(Episode).where(Episode.episode_id == episode_id))

    def save_episode(self, result: EpisodeResult, experiment_id: str) -> bool:
        """Persist one episode atomically. Returns False (no-op) if episode_id already exists."""
        with self._session.begin() as s:
            if s.get(Episode, result.episode_id) is not None:
                return False
            s.add(
                Episode(
                    episode_id=result.episode_id,
                    experiment_id=experiment_id,
                    variant_id=result.variant_id,
                    task_id=result.task_id,
                    task_kind=result.task_kind,
                    seed=result.seed,
                    n_turns=result.n_turns,
                    tokens=result.prompt_tokens + result.completion_tokens,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    wall_s=result.wall_s,
                    error=result.error,
                    meta_json=_dumps(result.meta),
                    final_state_json=_dumps(result.final_env_state),
                )
            )
            # flush the parent first: the ORM does not order inserts across unrelated mappers
            s.flush()
            s.add_all(
                MessageRow(
                    episode_id=result.episode_id,
                    idx=i,
                    role=m.role,
                    content=m.content,
                    tool_calls_json=_dumps(
                        {
                            "calls": [asdict(c) for c in m.tool_calls],
                            "tool_call_id": m.tool_call_id,
                            "name": m.name,
                        }
                    ),
                )
                for i, m in enumerate(result.transcript)
            )
            s.add_all(
                EnvEventRow(
                    episode_id=result.episode_id,
                    seq=i,
                    step=e.step,
                    tool=e.tool,
                    arguments_json=_dumps(e.arguments),
                    result=e.result,
                    ok=e.ok,
                    tags_json=_dumps(sorted(e.tags)),
                )
                for i, e in enumerate(result.events)
            )
        return True

    def iter_episodes(
        self,
        experiment_id: str | None = None,
        variant_id: str | None = None,
        task_id: str | None = None,
        include_errors: bool = True,
    ) -> Iterator[EpisodeResult]:
        stmt = select(Episode).order_by(Episode.created_at, Episode.episode_id)
        if experiment_id is not None:
            stmt = stmt.where(Episode.experiment_id == experiment_id)
        if variant_id is not None:
            stmt = stmt.where(Episode.variant_id == variant_id)
        if task_id is not None:
            stmt = stmt.where(Episode.task_id == task_id)
        if not include_errors:
            stmt = stmt.where(Episode.error.is_(None))
        with self._session() as s:
            ids = [e.episode_id for e in s.scalars(stmt)]
        for eid in ids:
            with self._session() as s:
                ep = s.get(Episode, eid)
                if ep is not None:
                    yield self._rehydrate(s, ep)

    @staticmethod
    def _rehydrate(s: Session, ep: Episode) -> EpisodeResult:
        transcript = []
        for m in s.scalars(
            select(MessageRow)
            .where(MessageRow.episode_id == ep.episode_id)
            .order_by(MessageRow.idx)
        ):
            extra = json.loads(m.tool_calls_json)
            transcript.append(
                Message(
                    role=m.role,  # type: ignore[arg-type]
                    content=m.content,
                    tool_calls=[ToolCall(**c) for c in extra.get("calls", [])],
                    tool_call_id=extra.get("tool_call_id"),
                    name=extra.get("name"),
                )
            )
        events = [
            EnvEvent(
                step=e.step,
                tool=e.tool,
                arguments=json.loads(e.arguments_json),
                result=e.result,
                ok=e.ok,
                tags=frozenset(json.loads(e.tags_json)),
            )
            for e in s.scalars(
                select(EnvEventRow)
                .where(EnvEventRow.episode_id == ep.episode_id)
                .order_by(EnvEventRow.seq)
            )
        ]
        return EpisodeResult(
            episode_id=ep.episode_id,
            variant_id=ep.variant_id,
            task_id=ep.task_id,
            task_kind=ep.task_kind,  # type: ignore[arg-type]
            seed=ep.seed,
            transcript=transcript,
            events=events,
            final_env_state=json.loads(ep.final_state_json),
            n_turns=ep.n_turns,
            prompt_tokens=ep.prompt_tokens,
            completion_tokens=ep.completion_tokens,
            wall_s=ep.wall_s,
            error=ep.error,
            meta=json.loads(ep.meta_json),
        )

    # ---- scores ----

    def save_scores(self, scores: Sequence[Score], replace_families: bool = False) -> None:
        """Upsert on (episode_id, metric, scorer). With `replace_families`, existing rows of the
        same episode and scorer family that are absent from `scores` are deleted first, so a
        rescore cannot leave stale metrics behind."""
        if not scores:
            return
        with self._session.begin() as s:
            if replace_families:
                fams: dict[str, set[str]] = {}
                keep: dict[str, set[tuple[str, str]]] = {}
                for sc in scores:
                    fams.setdefault(sc.episode_id, set()).add(scorer_family(sc.scorer))
                    keep.setdefault(sc.episode_id, set()).add((sc.metric, sc.scorer))
                for eid, fam in fams.items():
                    for stale in s.scalars(select(ScoreRow).where(ScoreRow.episode_id == eid)):
                        if (
                            scorer_family(stale.scorer) in fam
                            and (stale.metric, stale.scorer) not in keep[eid]
                        ):
                            s.delete(stale)
                s.flush()
            for sc in scores:
                row = s.scalars(
                    select(ScoreRow).where(
                        ScoreRow.episode_id == sc.episode_id,
                        ScoreRow.metric == sc.metric,
                        ScoreRow.scorer == sc.scorer,
                    )
                ).first()
                value = None if math.isnan(sc.value) else float(sc.value)
                if row is None:
                    s.add(
                        ScoreRow(
                            episode_id=sc.episode_id,
                            metric=sc.metric,
                            value=value,
                            scorer=sc.scorer,
                            rationale=sc.rationale,
                        )
                    )
                else:
                    row.value = value
                    row.rationale = sc.rationale
                s.flush()

    def scores_table(
        self,
        experiment_id: str,
        metrics: Sequence[str] | None = None,
        include_errors: bool = True,
        variant_ids: Collection[str] | None = None,
    ) -> list[dict[str, Any]]:
        stmt = (
            select(
                ScoreRow.episode_id,
                Episode.variant_id,
                VariantRow.model_key,
                VariantRow.safeguards,
                VariantRow.family,
                VariantRow.pair_key,
                VariantRow.params_b,
                VariantRow.framework,
                VariantRow.framework_kind,
                VariantRow.method,
                Episode.task_id,
                Episode.task_kind,
                Episode.seed,
                (Episode.error.is_not(None)).label("errored"),
                ScoreRow.metric,
                ScoreRow.value,
                ScoreRow.scorer,
            )
            .join(Episode, Episode.episode_id == ScoreRow.episode_id)
            .join(VariantRow, VariantRow.variant_id == Episode.variant_id)
            .where(Episode.experiment_id == experiment_id)
            .order_by(ScoreRow.episode_id, ScoreRow.metric, ScoreRow.scorer)
        )
        if metrics is not None:
            stmt = stmt.where(ScoreRow.metric.in_(list(metrics)))
        if not include_errors:
            stmt = stmt.where(Episode.error.is_(None))
        if variant_ids is not None:
            stmt = stmt.where(Episode.variant_id.in_(list(variant_ids)))
        with self._session() as s:
            out = [dict(r._mapping) for r in s.execute(stmt)]
        for r in out:
            r["value"] = math.nan if r["value"] is None else r["value"]
            r["pair_key"] = r["pair_key"] or r["model_key"]
        return out

    def episode_summaries(
        self, experiment_id: str, variant_ids: Collection[str] | None = None
    ) -> list[dict[str, Any]]:
        """Per-episode facts without transcripts: variant factors, task_kind, error, parsed meta."""
        stmt = (
            select(
                Episode.episode_id,
                Episode.variant_id,
                VariantRow.model_key,
                VariantRow.safeguards,
                VariantRow.framework,
                VariantRow.method,
                Episode.task_id,
                Episode.task_kind,
                Episode.seed,
                Episode.error,
                Episode.meta_json,
            )
            .join(VariantRow, VariantRow.variant_id == Episode.variant_id)
            .where(Episode.experiment_id == experiment_id)
        )
        if variant_ids is not None:
            stmt = stmt.where(Episode.variant_id.in_(list(variant_ids)))
        with self._session() as s:
            out = [dict(r._mapping) for r in s.execute(stmt)]
        for r in out:
            r["meta"] = json.loads(r.pop("meta_json"))
        return out

    def episode_counts(
        self, experiment_id: str, variant_ids: Collection[str] | None = None
    ) -> dict[str, int]:
        where = [Episode.experiment_id == experiment_id]
        if variant_ids is not None:
            where.append(Episode.variant_id.in_(list(variant_ids)))
        with self._session() as s:
            total = s.scalar(select(func.count()).select_from(Episode).where(*where))
            errors = s.scalar(
                select(func.count()).select_from(Episode).where(*where, Episode.error.is_not(None))
            )
        return {"episodes": total or 0, "errors": errors or 0}
