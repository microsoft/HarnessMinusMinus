# Contributor instructions

Keep this package importable without CUDA, model checkpoints, network access, or benchmark dependencies. Preserve frozen protocol identifiers and fallback behavior when changing public Python APIs.

Local validation is limited to static checks and synthetic unit tests. Run every benchmark and experiment on an authorized server, using a new run directory and complete provenance. Inspect GPU state, select a physical device, and record CUDA_VISIBLE_DEVICES before GPU runs.

Do not commit credentials, datasets, model weights, raw provider logs, internal host addresses, or historical experiment directories. Keep models and harnesses behind caller-supplied adapters.

Do not claim benchmark gains from unit tests or incomplete comparisons. Preserve negative results and uncertainty. Write concise, factual documentation.

Use SSH for GitHub Git transport.
