<!-- BEGIN MICROSOFT SECURITY.MD V1.0.0 BLOCK -->

## Security

Microsoft takes the security of our software products and services seriously, which
includes all source code repositories in our GitHub organizations.

**Please do not report security vulnerabilities through public GitHub issues.**

For security reporting information, locations, contact information, and policies,
please review the latest guidance for Microsoft repositories at
[https://aka.ms/SECURITY.md](https://aka.ms/SECURITY.md).

<!-- END MICROSOFT SECURITY.MD BLOCK -->

# Publication and security

Credentials belong in the integrating application's environment or secret store. Supply model transport and tool execution through callbacks. Never place a credential in code, examples, prompts, configuration committed to Git, or provider receipts intended for publication.

Keep benchmark data, learned snapshots, model weights, raw provider logs, user conversations, cluster configuration, and internal host details outside this repository. The source manifest uses repository-relative paths and hashes. Synthetic tests use invented identifiers.

Before a release, run:

```bash
python tools/check_release.py --history
python -m pytest -q
python -m build
python tools/check_release.py --distributions dist
```

The release check rejects common credential formats, embedded authentication values, personal home paths, private network addresses, unexpected file types, and common sensitive filenames. It checks extracted-source hashes against the source manifest. `--history` also scans reachable Git blobs; `--distributions` examines wheel and source archive contents. Pattern checks cannot establish the absence of every possible secret or personal detail; review the diff and archive inventory before publishing.

Report suspected vulnerabilities using the Microsoft process in the repository [Microsoft security guidance](https://aka.ms/SECURITY.md). Keep credential values and private records out of public issues. If a live credential is exposed, its owner must revoke or rotate it; removing the current file alone leaves earlier copies intact.
