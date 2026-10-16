"""Check public source, reachable Git blobs, and built distributions before release."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "credential format": re.compile(
        r"(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
        r"|AKIA[0-9A-Z]{16}|-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----)"
    ),
    "embedded credential": re.compile(
        r"(?i)(?:api[_-]?key|client[_-]?secret|password|access[_-]?token|auth[_-]?token)"
        r"\s*['\"]?\s*[:=]\s*['\"]([^'\"\r\n]+)['\"]"
    ),
    "authorization value": re.compile(r"(?i)\b(?:Bearer|Basic)\s+[A-Za-z0-9+/_.=-]{12,}"),
    "URL credentials": re.compile(r"https?://[^/\s:@]+:[^/\s@]+@"),
    "personal home path": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+"),
    "internal deployment detail": re.compile(r"\b(?:h200|h20)-\d+\b|/(?:mnt/phiagent-data|data[0-9])/"),
    "private network address": re.compile(
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}"
        r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
    ),
    "drafting residue": re.compile(r"(?i)as\s+an\s+AI\b|generated\s+by\s+chatgpt\b"),
}
EMAIL = re.compile(r"[A-Za-z0-9_.+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
SAFE_EMAIL_DOMAINS = {"example.com", "example.org", "example.net", "users.noreply.github.com"}
SUFFIXES = {".py", ".mjs", ".md", ".json", ".toml", ".cfg", ".in", ".yml", ".yaml", ".txt", ".csv"}
BARE_FILES = {"LICENSE", "NOTICE", "PKG-INFO", "WHEEL", "METADATA", "RECORD", "AGENTS.md", ".gitignore"}
BLOCKED_PARTS = {".ssh", ".aws", ".azure", ".venv", "credentials", "secrets", "__pycache__"}


def git(*args, input=None):
    return subprocess.check_output(["git", *args], cwd=ROOT, input=input)


def check_name(name, errors):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        errors.append(f"{name}: unsafe path")
    if any(part.lower() in BLOCKED_PARTS for part in path.parts):
        errors.append(f"{name}: private or generated directory")
    if path.name.startswith(".env") or path.suffix.lower() in {".pem", ".key", ".p12", ".pfx", ".p8", ".db"}:
        errors.append(f"{name}: sensitive filename")
    if path.name not in BARE_FILES and path.suffix.lower() not in SUFFIXES:
        errors.append(f"{name}: unexpected public file type")


def check_bytes(name, data, errors):
    check_name(name, errors)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        errors.append(f"{name}: non-text public asset requires review")
        return
    for number, line in enumerate(text.splitlines(), 1):
        for label, pattern in PATTERNS.items():
            if pattern.search(line):
                errors.append(f"{name}:{number}: {label}")
        for match in EMAIL.finditer(line.replace("git@github.com:", "")):
            if match[1].lower() not in SAFE_EMAIL_DOMAINS:
                errors.append(f"{name}:{number}: personal contact address")


def source_files():
    return sorted(set(git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")) - {""})


def check_source(errors):
    files = source_files()
    for name in files:
        path = ROOT / name
        if path.is_symlink():
            errors.append(f"{name}: symlink requires review")
        elif path.is_file():
            check_bytes(name, path.read_bytes(), errors)
        else:
            errors.append(f"{name}: missing published file")
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    for row in manifest["files"]:
        name = row["destination"]
        if name not in files:
            errors.append(f"{name}: source mapping is outside the release")
            continue
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != row["release_sha256"]:
            errors.append(f"{name}: release hash differs from source manifest")
        if not re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"]):
            errors.append(f"{name}: invalid source hash")
    print(f"Checked {len(files)} public files and {len(manifest['files'])} source mappings")


def check_history(errors):
    prefix = git("rev-parse", "--show-prefix").decode().strip()
    objects = git("rev-list", "--objects", "--all").decode().splitlines()
    names = {line.split(" ", 1)[0]: line.split(" ", 1)[1] for line in objects if " " in line}
    if prefix:
        names = {oid: name[len(prefix):] for oid, name in names.items() if name.startswith(prefix)}
    query = "".join(oid + "\n" for oid in names).encode()
    metadata = git("cat-file", "--batch-check=%(objectname) %(objecttype)", input=query).decode().splitlines()
    blobs = [line.split()[0] for line in metadata if line.endswith(" blob")]
    stream = io.BytesIO(git("cat-file", "--batch", input="".join(oid + "\n" for oid in blobs).encode()))
    for oid in blobs:
        header = stream.readline().decode().split()
        data = stream.read(int(header[2]))
        if stream.read(1) != b"\n":
            raise ValueError("Malformed Git object stream")
        check_bytes(f"history/{oid[:12]}/{names[oid]}", data, errors)
    print(f"Checked {len(blobs)} reachable historical blobs")


def check_distributions(directory, errors):
    archives = sorted(Path(directory).glob("*.whl")) + sorted(Path(directory).glob("*.tar.gz"))
    if not archives:
        errors.append("No built distributions found")
    for path in archives:
        count = 0
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as archive:
                for entry in archive.infolist():
                    if not entry.is_dir():
                        check_bytes(f"{path.name}/{entry.filename}", archive.read(entry), errors)
                        count += 1
        else:
            with tarfile.open(path) as archive:
                for entry in archive.getmembers():
                    if entry.issym() or entry.islnk():
                        errors.append(f"{path.name}/{entry.name}: archive link requires review")
                    elif entry.isfile():
                        check_bytes(f"{path.name}/{entry.name}", archive.extractfile(entry).read(), errors)
                        count += 1
        print(f"Checked {path.name}: {count} archived files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--distributions", type=Path)
    args = parser.parse_args()
    errors = []
    check_source(errors)
    if args.history:
        check_history(errors)
    if args.distributions:
        check_distributions(args.distributions, errors)
    for error in sorted(set(errors)):
        print(error, file=sys.stderr)
    if errors:
        return 1
    print("Publication checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
