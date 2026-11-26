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
        r"(?:(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
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
SUFFIXES = {".py", ".mjs", ".js", ".ts", ".bib", ".svg", ".png", ".gif", ".md", ".json", ".toml", ".cfg", ".in", ".yml", ".yaml", ".txt", ".csv"}
BARE_FILES = {"LICENSE", "NOTICE", "PKG-INFO", "WHEEL", "METADATA", "RECORD", "AGENTS.md", ".gitignore", ".mailmap", "LICENSE-MICROSOFT"}
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


# These published figures were reviewed as binary assets. A new image needs review.
FIGURE_HASHES = {
    "ablation.png": "1f31d8536ba1f2362694e0c1975b61ac005b8fb10ac2b07cb62e0c0dec90c453",
    "benchmark-results.png": "3eb48d0061921506dff43422d8dc01582f0e6b3906e407540881776ec8afef8c",
    "competitor-results.png": "43e21aa33ac3e5013873888f2ed788570c0337dd058e50364ac1ee0aad735e10",
    "guarded-reuse-overview.gif": "33b9cdb1ed8c6607697305a028320755d92f22401b7226c0e2055f22d202ff2a",
    "guarded-reuse.gif": "bcf1ea91179cd74774d4209a7bb4f5ec4e3d1139b994f1c7af15e2a32315bc8c",
    "guarded-reuse.png": "c15458f867af470188614e73b8e5bcd0dde865df52f404c801e7ecffcfda5362",
    "harness-overview-domain.png": "5cb5986abb3cb83dcfc9d141f6e6d80335388bdd4d6c37fe459e6ee3a4144ac1",
    "harnessmm-coding-harness-results.png": "2a305b8a831bb066f00c951d71bf6c528d911a05aea11a06f482a2ff836d9512",
    "harnessmm-six-benchmark-results.png": "375f8d0990cca1e8c2523e75745f5107288388a2a912f6579a77216ea341462b",
    "overview.png": "6a7b4b2965feb6cc1de458b762b4542ac37fee904526b50a755e551926c18ff1",
    "paper-figure-1.png": "8390377bf5db52cf5fbc0dbb39ba22a2bf0bd69b3118e13a8ef47b4755925202",
    "rsi-results.gif": "fe3f8d907cef027a2c0c1178670c9401d8877d71900245fd8666d6b9cb2f969d",
    "tsinghua-emblem.png": "735fb1b0df6f233ded7ba755efca058181d6ce1b843fe53da8f68c1ac603a190",
}

# Preserve the existing author alias map; changes still require a fresh review.
MAILMAP_SHA256 = "383cfc8ebb47104e67007a07d47fcbf1a1e1adb1c55f85869a6b0954e2a3c5b9"


def public_example(name, label, match):
    path = PurePosixPath(name)
    if label == "private network address":
        # Docker's default bridge gateway and its documented private address ranges.
        return match[0] in {"172." + "17.0.1", "172." + "16.0.0", "192." + "168.0.0"} and path.suffix == ".py"
    if label == "embedded credential":
        if match[1] == "EMPTY":
            return "scripts" in path.parts
        return ("tests" in path.parts or "test" in path.parts) and match[1] in {
            "fake", "synthetic", "synthetic-not-a-credential", "hmm-test", "y", "z"
        }
    if label == "authorization value":
        return path.name == "deepseek_harness_plugin.test.mjs" and match[0] == "Bearer " + "SYNTHETIC_BRIDGE_SECRET"
    return False


HISTORICAL_FIGURE_HASHES = {
    "ablation.png": {"5a5e2b99f4a82c78af6b8c69a2b846db256da04c7693f4cf8c574b39858ec7ff"},
    "benchmark-results.png": {"56b06a6bf0ecfa32a3714ad8d6a67f55426b5cbab3e092e6a95d9143456c4c08"},
    "competitor-results.png": {"7814f0cda9f4aa2b72806ac898511c1c140e51d67466e23b06bc74b6a430353b"},
    "guarded-reuse.gif": {"62647e7bd0edfbb43f28d3c17e120e35935094bb9245307fffe66c78cef3fcf9", "e32930a047583552f8dc1267503e693ea3460179a6d941fece10d29564b4f797"},
    "guarded-reuse.png": {"8503e05dfdc0d32f25982a54a972aee706b398d4c38dca431d80f8612793a57a", "9ed0770e8a721c8e6a847f082443bc3a4d47a10e3edc4b3445b44cbb7bfa8fe4"},
    "paper-figure-1.png": {"9c13a7072429132ef2f3ad23b2e434d97e9b99a051349a206c198a59878b2e7e"},
}


def check_bytes(name, data, errors, *, historical=False):
    check_name(name, errors)
    path = PurePosixPath(name)
    if path.name == ".mailmap":
        if hashlib.sha256(data).hexdigest() != MAILMAP_SHA256:
            errors.append(f"{name}: author alias map requires review")
        return
    if path.suffix in {".gif", ".png"}:
        digest = hashlib.sha256(data).hexdigest()
        reviewed = digest == FIGURE_HASHES.get(path.name)
        if historical and digest in HISTORICAL_FIGURE_HASHES.get(path.name, ()):
            reviewed = True
        if (path.parent.parts[-3:] != ("docs", "assets", "readme")
                or not reviewed):
            errors.append(f"{name}: unreviewed binary asset")
        return
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        errors.append(f"{name}: non-text public asset requires review")
        return
    for number, line in enumerate(text.splitlines(), 1):
        for label, pattern in PATTERNS.items():
            if any(not public_example(name, label, match) for match in pattern.finditer(line)):
                errors.append(f"{name}:{number}: {label}")
        for match in EMAIL.finditer(line.replace("git@github.com:", "").replace("ssh://git@github.com/", "")):
            if match[1].lower() not in SAFE_EMAIL_DOMAINS and match[0] != "opencode@microsoft.com":
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
        name = row["path"]
        if name not in files:
            errors.append(f"{name}: source mapping is outside the release")
            continue
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != row["published_sha256"]:
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
        check_bytes(f"history/{oid[:12]}/{names[oid]}", data, errors, historical=True)
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
