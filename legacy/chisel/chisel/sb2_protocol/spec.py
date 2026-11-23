"""Frozen identities and finite policies for the SB2 five-method protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

OFFICIAL_CODE_COMMIT = "5c160265aa93c15b38e4034cbf1e09ab498335d9"
OFFICIAL_DATA_REVISION = "9dea60025792fbac5928ce9f44812362dccbeecd"
OFFICIAL_ARCHIVE_SHA256 = "17147ef9578cd57ce76c9a719d19da7821f3e5cb0d8f776c820f699fdcdb761c"
SKILLOPT_REVISION = "eb8c1e7bcbccdd80f9d422f12018fcd8e84ce19a"
AUTOSADDLER_REVISION = "30e20ce004486c58e7ee97c66182a8d0d41ec90e"

CATEGORIES = ("Debugging", "Financial_Model", "Template")
SPLIT_COUNTS = {
    "development": {"Debugging": 30, "Financial_Model": 35, "Template": 32},
    "confirmation": {"Debugging": 30, "Financial_Model": 35, "Template": 35},
    "test": {"Debugging": 40, "Financial_Model": 30, "Template": 30},
}
FORMAL_DEVELOPMENT_COUNTS = {
    "Debugging": 0,
    "Financial_Model": 35,
    "Template": 32,
}
METHOD_LABELS = {
    "no_skill": "No-Skill (real official SB2 agent baseline)",
    "skillopt": "SkillOpt core + PhiAgent SB2 adapter (no official SB2 checkpoint)",
    "autosaddler": "official-public AutoSaddler V2 engine + PhiAgent SB2 scenario adapter",
    "c_gspo": "finite constrained-GSPO policy",
    "ours": "C-GSPO + auditable ADD/DELETE/MODIFY structural preflight",
}
FORMAL_METHODS = ("no_skill", "skillopt", "autosaddler", "ours")
INTERNAL_DEVELOPMENT_CONTROLS = ("c_gspo",)
PRIMARY_METRIC = "accuracy"
PRIMARY_METRIC_LABEL = "Acc."
PROTOCOL_SCHEMA = "phiagent-spreadsheetbench2-five-method/v1"

CGSPO_POLICY: dict[str, Any] = {
    "algorithm": "finite tabular constrained GSPO-style sequence policy",
    "candidate_count": 4,
    "candidate_ids": [
        "full_verification",
        "typed_minimal",
        "formula_structure",
        "template_preserving",
    ],
    "policy_epochs": 4,
    "learning_rate": 0.25,
    "clip_epsilon": 0.2,
    "lagrange_learning_rate": 1.0,
    "initial_lagrange_multiplier": 4.0,
    "minimum_accuracy_retention": 1.0,
    "maximum_updates": 1,
    "selection": (
        "highest final policy probability after masking every candidate that loses any "
        "No-Skill development success; deterministic ties use lower tokens then candidate_id"
    ),
}

CGSPO_CANDIDATES: dict[str, tuple[str, ...]] = {
    "full_verification": (
        "Inspect workbook structure and relevant peer formulas before editing.",
        "Use the smallest task-supported edit and preserve unrelated workbook content.",
        "Check formula references, signs, ranges, units, styles, and period alignment.",
        "Reopen and verify the saved workbook before submitting.",
    ),
    "typed_minimal": (
        "Classify each intended edit as formula, literal, style, or structure.",
        "Make only edits necessary for the instruction and preserve all unrelated content.",
        "Reopen and verify the saved workbook before submitting.",
    ),
    "formula_structure": (
        "Infer formulas from row, column, cross-sheet, and period structure; never guess a value.",
        "Check references, signs, ranges, units, and relative/absolute anchors.",
        "Preserve unrelated workbook content and verify the saved workbook.",
    ),
    "template_preserving": (
        "Extend the nearest structurally matching template pattern.",
        "Preserve styles, merged cells, sheets, formulas, and unrelated values.",
        "Reopen and verify the saved workbook before submitting.",
    ),
}


class ProtocolError(RuntimeError):
    """A fail-closed protocol or infrastructure error."""


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def verify_self_hash(value: Mapping[str, Any], field: str = "canonical_sha256") -> None:
    unsigned = {key: item for key, item in value.items() if key != field}
    if value.get(field) != canonical_sha256(unsigned):
        raise ProtocolError(f"self-hash mismatch: {field}")


def validate_items(rows: object, split: str) -> list[dict[str, Any]]:
    expected_counts = {
        **SPLIT_COUNTS,
        "formal_development": FORMAL_DEVELOPMENT_COUNTS,
    }
    if split not in expected_counts or not isinstance(rows, list):
        raise ProtocolError(f"invalid {split} item manifest")
    expected = expected_counts[split]
    required = {
        "id",
        "instruction",
        "spreadsheet_path",
        "golden_response_path",
        "answer_position",
        "category",
        "task_key",
        "workbook_group",
    }
    keys: set[str] = set()
    groups: set[str] = set()
    counts = {category: 0 for category in CATEGORIES}
    checked: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict) or not required <= set(raw):
            raise ProtocolError(f"{split} item schema differs")
        row = dict(raw)
        category = str(row["category"])
        key = str(row["task_key"])
        group = f"{category}:{row['workbook_group']}"
        if category not in counts or key in keys:
            raise ProtocolError(f"{split} item identity differs: {key}")
        if not key.startswith(f"{category}:"):
            raise ProtocolError(f"unqualified task key: {key}")
        counts[category] += 1
        keys.add(key)
        groups.add(group)
        checked.append(row)
    if counts != expected or len(checked) != sum(expected.values()):
        raise ProtocolError(f"{split} counts differ: {counts}")
    return checked


def method_contracts() -> dict[str, dict[str, Any]]:
    return {
        "no_skill": {
            "label": METHOD_LABELS["no_skill"],
            "agent": "official SpreadsheetBench 2 SWE-agent scaffold",
            "skill": None,
            "real_inference_required": True,
            "missing_or_no_edit_output_scored_zero": True,
            "structural_preflight_failure_scored_zero": True,
            "post_output_verification_failure_scored_zero": True,
            "missing_output_is_never_fabricated": True,
        },
        "skillopt": {
            "label": METHOD_LABELS["skillopt"],
            "source_revision": SKILLOPT_REVISION,
            "checkpoint": None,
            "adapter": "official SB2 scaffold plus pinned SkillOpt spreadsheet core",
            "identity_caveat": "Official SkillOpt publishes no SpreadsheetBench 2 checkpoint.",
        },
        "autosaddler": {
            "label": METHOD_LABELS["autosaddler"],
            "engine_revision": AUTOSADDLER_REVISION,
            "engine": "official-public complete AutoSaddler V2",
            "adapter": "phiagent_autosaddler.spreadsheetbench2_scenario",
            "acceptance": "matched valid strict improvement; full development on accept",
            "budget": {"max_iterations": 4, "max_rollouts": 256},
        },
        "c_gspo": {
            "label": METHOD_LABELS["c_gspo"],
            "policy": CGSPO_POLICY,
            "candidates": {key: list(value) for key, value in CGSPO_CANDIDATES.items()},
        },
        "ours": {
            "label": METHOD_LABELS["ours"],
            "base_policy": CGSPO_POLICY,
            "preflight": "OOXML ADD/DELETE/MODIFY diff before outcome access",
            "input_only_router": {
                "candidate_when": "materialized cells >= 60 and unique styles >= 19",
                "otherwise": "no_skill",
            },
            "sparse_execution_gate": {
                "category": "Template",
                "formula_count": 0,
                "materialized_cell_count": [55, 250],
                "minimum_unique_styles": 13,
                "otherwise": "skip and score task failure as zero",
            },
            "repair_routes": [
                "accept_verified_candidate",
                "full_no_skill_baseline_fallback",
                "fail_closed",
            ],
            "financial_template_fallback": "unsupported OR confidence < 1.0",
            "recalculation": "LibreOffice headless followed by OOXML post-output verification",
        },
    }
