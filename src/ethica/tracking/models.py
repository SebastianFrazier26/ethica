"""SQLAlchemy 2.0 typed models for the tracking store (DECISIONS D-11)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy import UniqueConstraint as Unique
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    git_rev: Mapped[str | None] = mapped_column(String, nullable=True)


class ExperimentRun(Base):
    """One row per phase invocation, so config edits between runs stay auditable."""

    __tablename__ = "experiment_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    phase: Mapped[str] = mapped_column(String, nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (Index("ix_experiment_runs_experiment", "experiment_id"),)


class Variant(Base):
    __tablename__ = "variants"

    variant_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_key: Mapped[str] = mapped_column(String, nullable=False)
    hf_repo: Mapped[str] = mapped_column(String, nullable=False)
    safeguards: Mapped[str] = mapped_column(String, nullable=False)
    family: Mapped[str] = mapped_column(String, nullable=False)
    # base model key for stock/abliterated pairing; own key for stock models
    pair_key: Mapped[str | None] = mapped_column(String, nullable=True)
    params_b: Mapped[float] = mapped_column(Float, nullable=False)
    framework: Mapped[str] = mapped_column(String, nullable=False)
    framework_kind: Mapped[str] = mapped_column(String, nullable=False)
    method: Mapped[str] = mapped_column(String, nullable=False)
    adapter_path: Mapped[str | None] = mapped_column(String, nullable=True)
    spec_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    __table_args__ = (
        Index("ix_variants_factors", "model_key", "safeguards", "framework", "method"),
    )


class Episode(Base):
    __tablename__ = "episodes"

    episode_id: Mapped[str] = mapped_column(String, primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    variant_id: Mapped[str] = mapped_column(ForeignKey("variants.variant_id"), nullable=False)
    task_id: Mapped[str] = mapped_column(String, nullable=False)
    task_kind: Mapped[str] = mapped_column(String, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    n_turns: Mapped[int] = mapped_column(Integer, nullable=False)
    tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    # tokens == prompt_tokens + completion_tokens; split kept so rehydration is lossless.
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    wall_s: Mapped[float] = mapped_column(Float, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    final_state_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (
        Index("ix_episodes_experiment_variant", "experiment_id", "variant_id"),
        Index("ix_episodes_experiment_task", "experiment_id", "task_id", "seed"),
        Index("ix_episodes_variant", "variant_id"),
    )


class MessageRow(Base):
    __tablename__ = "messages"

    episode_id: Mapped[str] = mapped_column(
        ForeignKey("episodes.episode_id", ondelete="CASCADE"), primary_key=True
    )
    idx: Mapped[int] = mapped_column(Integer, primary_key=True)
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # JSON object: {"calls": [...], "tool_call_id": str|null, "name": str|null}
    tool_calls_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")


class EnvEventRow(Base):
    __tablename__ = "env_events"

    episode_id: Mapped[str] = mapped_column(
        ForeignKey("episodes.episode_id", ondelete="CASCADE"), primary_key=True
    )
    # position in the event list; EnvEvent.step is not guaranteed unique per episode
    seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    tool: Mapped[str] = mapped_column(String, nullable=False)
    arguments_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    result: Mapped[str] = mapped_column(Text, nullable=False, default="")
    ok: Mapped[bool] = mapped_column(Boolean, nullable=False)
    tags_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")


class ScoreRow(Base):
    __tablename__ = "scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    episode_id: Mapped[str] = mapped_column(
        ForeignKey("episodes.episode_id", ondelete="CASCADE"), nullable=False
    )
    metric: Mapped[str] = mapped_column(String, nullable=False)
    # NaN is stored as NULL (SQLite has no NaN) and restored on read
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    scorer: Mapped[str] = mapped_column(String, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False, default="")

    __table_args__ = (
        Unique("episode_id", "metric", "scorer", name="uq_scores_episode_metric_scorer"),
        Index("ix_scores_metric", "metric"),
    )
