"""Synthetic Git provenance checks; no benchmark data or network calls."""

import hashlib
import json
import subprocess

import pytest

from scripts import run_chisel_component_quality as quality
from scripts import run_searchqa_residual_meta_campaign as searchqa
from scripts.source_repository import repository_from_remote, source_repository


@pytest.mark.parametrize("remote", [
    "git@github.com:microsoft/HarnessMinusMinus.git",
    "ssh://git@github.com/microsoft/HarnessMinusMinus.git",
    "https://github.com/microsoft/HarnessMinusMinus.git",
    "https://github.com/microsoft/HarnessMinusMinus",
])
def test_supported_remote_forms(remote):
    assert repository_from_remote(remote) == "microsoft/HarnessMinusMinus"


@pytest.mark.parametrize("remote", [
    "", "../checkout", "https://example.com/owner/repo",
    "https://github.com/owner/repo/issues", "https://github.com/owner/..",
    "https://github.com/owner/repo?token=synthetic",
    "https://github.com/owner/repo#main", "git@github.com:owner/repo\n",
])
def test_unknown_or_malformed_remote_fails_closed(remote):
    with pytest.raises(ValueError):
        repository_from_remote(remote)


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    git(root, "init", "--initial-branch=main")
    git(root, "-c", "user.name=Synthetic", "-c", "user.email=synthetic@example.com",
        "commit", "--allow-empty", "-m", "Synthetic source")
    git(root, "remote", "add", "origin", "git@github.com:example/released-source.git")
    return root


def test_runners_record_actual_origin_and_detect_repository_changes(checkout, monkeypatch):
    monkeypatch.setattr(searchqa, "ROOT", checkout)
    monkeypatch.setattr(quality, "ROOT", checkout)
    assert source_repository(checkout) == "example/released-source"
    state = searchqa._git_state()
    assert state["repository"] == "example/released-source"
    assert state["head"] == git(checkout, "rev-parse", "HEAD")
    assert quality.git_provenance(checkout, None)["git_repository"] == state["repository"]
    git(checkout, "remote", "set-url", "origin", "git@github.com:example/fork.git")
    assert searchqa._git_state()["repository"] == "example/fork"


def test_missing_origin_never_silently_labels_source_as_phiagent(checkout, monkeypatch):
    monkeypatch.setattr(searchqa, "ROOT", checkout)
    git(checkout, "remote", "remove", "origin")
    with pytest.raises(subprocess.CalledProcessError):
        searchqa._git_state()


def test_external_receipt_identity_is_bound_and_original_bytes_preserved(
    checkout, tmp_path, monkeypatch,
):
    monkeypatch.setattr(searchqa, "ROOT", checkout)
    monkeypatch.setattr(quality, "ROOT", checkout)
    state = searchqa._git_state()
    receipt = tmp_path / "receipt.json"
    original = (json.dumps(state, indent=2) + "\n").encode()
    receipt.write_bytes(original)
    output = tmp_path / "run"
    provenance = quality.git_provenance(output, receipt)
    assert provenance["source_git_state"]["repository"] == "example/released-source"
    assert (output / "source-git-state.json").read_bytes() == original
    assert provenance["source_git_state_sha256"] == hashlib.sha256(original).hexdigest()

    state["repository"] = "example/historical-source"
    receipt.write_text(json.dumps(state, indent=2) + "\n")
    historical_bytes = receipt.read_bytes()
    with pytest.raises(quality.ProtocolError, match="invalid identity"):
        quality.git_provenance(tmp_path / "rejected", receipt)
    assert not (tmp_path / "rejected").exists()

    # An explicit historical identity works in source-only deployments without Git.
    monkeypatch.setattr(quality, "ROOT", tmp_path / "source-only")
    provenance = quality.git_provenance(
        tmp_path / "historical", receipt, expected_repository="example/historical-source",
    )
    assert provenance["source_git_state"] == state
    assert (tmp_path / "historical/source-git-state.json").read_bytes() == historical_bytes


def test_repository_override_cannot_relabel_local_source(checkout, monkeypatch):
    monkeypatch.setattr(quality, "ROOT", checkout)
    with pytest.raises(quality.ProtocolError, match="requires --source-git-state"):
        quality.git_provenance(checkout, None, expected_repository="example/other-source")


def test_component_source_freeze_includes_repository_helper():
    assert quality.ROOT / "scripts/source_repository.py" in quality.source_paths()
