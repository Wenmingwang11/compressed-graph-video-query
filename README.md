# Compressed Graph Video Query

Research code for structured video querying using compressed graph representations.

> **Submission status:** The associated manuscript is currently being submitted to **EDBT**. It has not been accepted. This is a submission-stage public release with selected core implementation temporarily withheld.

## Repository contents

- Root Python scripts: project entry points and utilities.
- `vsimsearch/`: graph/data utilities and interfaces; core modules currently contain explicit withheld placeholders.
- `baseline/`: local baseline integration code.
- `supplement/`: experiment, evaluation and visualization scripts.
- `tests/`: available research tests; tests exposing withheld internals are placeholders.
- `figures/`: figures referenced by the local manuscript; experimental figures remain drafts.
- `requirements.txt`: original project dependency specification.
- `release_manifest.json`: file-level release status and SHA-256 hashes.

## Availability and execution

See [RELEASE_SCOPE.md](RELEASE_SCOPE.md) for the precise withheld module list. **This release cannot run the complete indexing/query pipeline or fully reproduce the manuscript experiments.** Withheld modules raise `NotImplementedError` rather than silently returning fabricated results.

The public scripts are research code and may require local path adjustments, separately obtained datasets and original baseline installations. The original dependency specification is provided for reference; no fresh environment installation or full reproduction has been verified for this public package.

```bash
python -m venv .venv
# Activate the environment for your platform, then:
python -m pip install -r requirements.txt
```

Do not run complete indexing/query entry points until the withheld implementations are available. Dataset files, weights, private annotations and credentials are not included.

## Figures and results

Figures are provided as submission drafts. They must not be interpreted as final validated experimental results. The local manuscript figure source is recorded in the release manifest.

## Citation and licensing

Final bibliographic information and release licensing will be added when settled. No open-source license is granted by this initial partial release; any third-party code remains subject to its upstream terms.
