"""Read source repository identity from Git without rewriting historical receipts."""

from pathlib import Path
import re
import subprocess


def validate_repository(repository: str) -> str:
    """Require a GitHub owner/repository identity, not a URL or local path."""
    if not isinstance(repository, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+", repository
    ) or repository.split("/")[-1] in {".", ".."}:
        raise ValueError("source repository must be a GitHub owner/repository")
    return repository


def repository_from_remote(remote: str) -> str:
    """Accept the GitHub SSH and HTTPS remote forms used by source checkouts."""
    prefixes = ("git@github.com:", "ssh://git@github.com/", "https://github.com/")
    for prefix in prefixes:
        if remote.startswith(prefix):
            return validate_repository(remote[len(prefix):].removesuffix(".git"))
    raise ValueError("source origin must be a GitHub SSH or HTTPS remote")


def source_repository(root: Path) -> str:
    """Fail when origin is missing rather than claiming another project's identity."""
    remote = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], cwd=root, text=True,
        stderr=subprocess.PIPE, timeout=60,
    ).strip()
    return repository_from_remote(remote)
