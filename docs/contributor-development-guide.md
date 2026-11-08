# Contributing to the implemented package and its documentation

[Architecture](architecture/index.md) · [Testing](testing-guide.md) · [Root agent guide](../AGENTS.md)

## Getting Started

### Development Environment Setup

Clone the repository and select Python 3.13 explicitly for scientific work:

```bash
git clone https://github.com/docxology/symergetics.git
cd symergetics
uv sync --python 3.13 --extra scientific --extra test --extra dev
uv run --python 3.13 --extra scientific --extra test --extra dev python -c "import symergetics; print(symergetics.__version__)"
```

Do not rewrite the repository's `.python-version` merely to run the scientific
environment. Core metadata supports Python >=3.11, but that is not evidence
that all optional scientific wheels exist for every newer interpreter. See
[installation routes](../README.md#install-and-run). The unchanged 1.3.0 version
string alone does not identify unreleased main features; record the revision.

## Development Workflow

1. Read root/scoped `AGENTS.md` and locate the relevant source and existing consumers.
2. Reuse existing module and test patterns; define the mathematical units and
   quantization boundary before changing an API.
3. Implement the requested behavior and update all affected callers and docs.
4. Run the checks required for that assignment, then report actual commands,
   environment, outcomes, skipped checks and artifact paths. Never paste an old
   receipt as current evidence. Some coordinated passes intentionally defer
   integrated checks until edits settle; follow that assignment's instructions.
5. Keep user-generated `output/` and immutable release artifacts out of changes.

## Architecture Guidelines

`core/` supplies rational scalars and coordinates; `geometry/` supplies indexed
surfaces and transformations; `computation/` supplies finite number-pattern
analysis; `utils/` supplies conversion/mnemonic/reporting helpers; `visualization/`
produces files and metadata. The package has no runtime import dependency on
QuadMath or QuadCraft. See the [dependency map](architecture/index.md).

Preserve exact native arithmetic where available, state floating boundaries,
and do not treat a vertex average as a solid mass centroid. Standard Python
exceptions are used; there is no shipped `core/errors.py` custom exception tree.
A geometry diagnostic must not imply self-intersection detection it does not do.

## Testing Strategy

Tests should exercise uncertain consumer-visible behavior with real data:
geometry, resource closure, error transitions and integration boundaries.
Prefer exact reference calculations for exact APIs and declared tolerances for
floating results. Do not add tests that only pin prose, default words, source
text or export existence. Avoid replacing real computations with mocks to make
a gate green. Commands and coverage semantics are centralized in the
[testing guide](testing-guide.md).

## Documentation Standards

- Keep the [documentation index](index.md) navigable and use links to canonical
  API contracts instead of duplicating conflicting recipes.
- Show complete imports and actual signatures. Label application adapters and
  conceptual extensions instead of presenting them as installed methods.
- Name coordinate gauge, scale, volume units and quantization in numerical examples.
- Distinguish development checkout features from immutable release-wheel APIs.
- Keep benchmark evidence tied to a command, revision, input and environment.
- If a heading changes, update inbound fragment links; preserve explicit anchors.

## Continuous Integration

The actual workflows, rather than illustrative replacement YAML, are authoritative:
[package verification](../.github/workflows/verify.yml) and
[suite verification](../.github/workflows/suite.yml). Dependencies, extras, Python
requirements and pytest options are declared in [pyproject.toml](../pyproject.toml).
A partial local test run does not establish the full package coverage gate.

## Release Process

Distribution uses checkout installation or wheels downloaded from
[GitHub Releases](https://github.com/docxology/symergetics/releases), not PyPI.
Do not publish with `twine` or claim that a new main-only keyword is supported by
an existing wheel. Published 1.3.0 artifacts are immutable. Release/version/tag
changes require their own maintainer-approved process; this guide does not
authorize them. See [release notes](../CHANGELOG.md) and the
[suite reproducibility guide](synergetics-suite.md).
