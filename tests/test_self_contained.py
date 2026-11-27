"""Synthetic cross-checkout regressions for source-only release checks."""

import json

import pytest

from scripts.check_self_contained import check


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project.urls]\nRepository = "https://github.com/microsoft/HarnessMinusMinus"\n'
    )
    (root / "SOURCE_MANIFEST.json").write_text(json.dumps({"files": []}))
    return root


@pytest.mark.parametrize("location", ["README.md", "pi-harnessmm/NOTICE", "tests/fixture.md"])
def test_cross_checkout_links_are_rejected_in_every_source_location(source, location):
    target = source / location
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("https://github.com/" + "personal-owner/" + "HarnessMM\n")
    errors, _ = check(source)
    assert any(error.startswith(location + ": points to another project:") for error in errors)


@pytest.mark.parametrize("repository", [
    "personal-owner/HarnessMM", "personal-owner/HarnessMinusMinus",
    "another-owner/HarnessMM-R", "another-owner/HarnessMM-V",
    "another-owner/PhiAgent-0", "another-owner/pi-harnessmm",
    "another-owner/dsh-harnessmm",
])
def test_project_family_links_reject_any_external_owner(source, repository):
    (source / "README.md").write_text("https://github.com/" + repository)
    assert check(source)[0]


@pytest.mark.parametrize("prefix", [
    "https://github.com/", "git@github.com:", "ssh://git@github.com/",
    "git:github.com/", "git+https://github.com/",
    "https://raw.githubusercontent.com/", "https://api.github.com/repos/",
    "https://codeload.github.com/", "https://www.github.com/",
])
def test_project_links_cover_git_raw_archive_and_api_urls(source, prefix):
    (source / "README.md").write_text(prefix + "personal-owner/" + "HarnessMinusMinus.git\n")
    assert check(source)[0]


def test_canonical_and_official_third_party_sources_are_allowed(source):
    links = [
        "microsoft/HarnessMinusMinus", "ByteDance-Seed/EdgeBench",
        "GT-Vision-Lab/VQA", "ChaoningZhang/MobileSAM",
    ]
    (source / "README.md").write_text("\n".join("https://github.com/" + link for link in links))
    (source / "NOTICE").write_text("https://github.com/" + "jlon/" + "pi-harnessmm")
    assert check(source)[0] == []


def test_upstream_attribution_exception_cannot_redirect_installation(source):
    (source / "README.md").write_text("https://github.com/" + "jlon/" + "pi-harnessmm")
    assert check(source)[0]


def test_project_link_check_is_case_insensitive(source):
    (source / "README.md").write_text("HTTPS://GITHUB.COM/" + "Another-Owner/" + "HARNESSMM")
    assert check(source)[0]
