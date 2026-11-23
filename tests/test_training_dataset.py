import json

import pytest

from ethica.conditioning.lora import data_hash
from ethica.training.dataset import build_chat_dataset, common_prompt_ids


def _arm_file(path, arm, n=40, reject=()):
    with path.open("w") as fh:
        for i in range(n):
            row = {
                "prompt_id": f"p{i:02d}",
                "category": "advice",
                "prompt": f"Question {i}?",
                "answer": f"{arm} answer {i}",
                "finish_reason": "stop",
                "reject": "refusal" if i in reject else None,
                "arm": arm,
                "teacher": "t",
            }
            fh.write(json.dumps(row) + "\n")
    return path


def _load(path):
    return [json.loads(line)["messages"] for line in path.read_text().splitlines()]


def test_common_ids_drop_prompts_rejected_in_any_arm(tmp_path):
    a = _arm_file(tmp_path / "a.jsonl", "virtue", reject={3})
    b = _arm_file(tmp_path / "b.jsonl", "generic", reject={7})
    ids = common_prompt_ids([a, b])
    assert len(ids) == 38 and "p03" not in ids and "p07" not in ids
    assert ids == sorted(ids)


def test_arms_differ_only_in_answers(tmp_path):
    a = _arm_file(tmp_path / "a.jsonl", "virtue", reject={3})
    b = _arm_file(tmp_path / "b.jsonl", "generic", reject={7})
    ids = common_prompt_ids([a, b])
    ma = build_chat_dataset(a, tmp_path / "da", ids)
    mb = build_chat_dataset(b, tmp_path / "db", ids)
    assert (ma["n_train"], ma["n_valid"]) == (mb["n_train"], mb["n_valid"]) == (34, 4)
    for split in ("train", "valid"):
        xa, xb = (
            _load(tmp_path / "da" / f"{split}.jsonl"),
            _load(tmp_path / "db" / f"{split}.jsonl"),
        )
        assert [m[0] for m in xa] == [m[0] for m in xb]
        assert all(m[1]["content"].startswith("virtue") for m in xa)
        assert all(m[1]["content"].startswith("generic") for m in xb)
        assert all([t["role"] for t in m] == ["user", "assistant"] for m in xa)
    assert ma["arm"] == "virtue" and ma["data_hash"] == data_hash(tmp_path / "da")
    assert ma["data_hash"] != mb["data_hash"]


def test_rejected_rows_never_reach_the_dataset(tmp_path):
    a = _arm_file(tmp_path / "a.jsonl", "virtue", reject={0, 1, 2})
    build_chat_dataset(a, tmp_path / "d", common_prompt_ids([a]))
    text = (tmp_path / "d" / "train.jsonl").read_text() + (
        tmp_path / "d" / "valid.jsonl"
    ).read_text()
    assert "Question 0?" not in text and "Question 3?" in text


def test_missing_or_too_few_ids_raise(tmp_path):
    a = _arm_file(tmp_path / "a.jsonl", "virtue", reject={5})
    with pytest.raises(ValueError, match="1 requested prompts have no kept answer"):
        build_chat_dataset(a, tmp_path / "d", [f"p{i:02d}" for i in range(10)])
    with pytest.raises(ValueError, match="need at least 8 examples"):
        build_chat_dataset(a, tmp_path / "d", ["p00", "p01"])
    with pytest.raises(ValueError, match="no arm files"):
        common_prompt_ids([])


def test_shared_rows_are_identical_in_every_arm(tmp_path):
    shared = [
        {
            "id": f"tool-{i}",
            "messages": [
                {"role": "system", "content": "S"},
                {"role": "user", "content": f"task {i}"},
                {"role": "assistant", "content": "<done/>"},
            ],
        }
        for i in range(10)
    ]
    a = _arm_file(tmp_path / "a.jsonl", "virtue")
    b = _arm_file(tmp_path / "b.jsonl", "generic")
    ids = common_prompt_ids([a, b])
    ma = build_chat_dataset(a, tmp_path / "da", ids, shared_rows=shared)
    mb = build_chat_dataset(b, tmp_path / "db", ids, shared_rows=shared)
    assert (
        ma["n_shared"] == 10
        and ma["n_train"] + ma["n_valid"] == 50 == mb["n_train"] + mb["n_valid"]
    )
    for split in ("train", "valid"):
        xa, xb = (
            _load(tmp_path / "da" / f"{split}.jsonl"),
            _load(tmp_path / "db" / f"{split}.jsonl"),
        )
        assert [m for m in xa if m[0]["role"] == "system"] == [
            m for m in xb if m[0]["role"] == "system"
        ]
        assert [m[0]["role"] == "system" for m in xa] == [m[0]["role"] == "system" for m in xb]
    with pytest.raises(ValueError, match="shared row ids"):
        build_chat_dataset(a, tmp_path / "d", ids, shared_rows=[shared[0], shared[0]])
