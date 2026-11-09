# Symergetics documentation: implementation, mathematics and research context

The development checkout supplies rational arithmetic, integer quadray geometry,
finite number-pattern analysis and visualization. Four labels represent three
spatial dimensions, not four independent axes. Conceptual applications are
separated below from implemented package APIs.

Repository archive DOI: [10.5281/zenodo.17114383](https://zenodo.org/records/17114383).
Paper archive DOI: [10.5281/zenodo.17114390](https://zenodo.org/records/17114390).
These identify archived records, not evidence that current main is a release.

## Start here

1. [Install a checkout or release wheel](../README.md#install-and-run) — not PyPI;
   scientific commands explicitly select Python 3.13.
2. [Core concepts](core-concept-guide.md) and [first calculations](tutorials/index.md).
3. [API contracts](api/index.md), [cookbook](api-cookbook-recipes.md) and [runnable demos](examples/index.md).
4. [Scientific poster sections](api/index.md#scientific-poster-sections)
   — focused default versus all seventeen topics; selector introduced in 1.4.0.
5. [Suite conventions and executable integration](synergetics-suite.md).

## Mathematics and implementation

- [Quadray quotient, metrics and volume](mathematical-foundations-synergetics.md).
- [FCC neighborhoods and scale-declared polyhedra](geometric-patterns-spatial-relationships.md).
- [Glossary: API versus conceptual terms](glossary-terminology.md).
- [Advanced numerical and surface boundaries](advanced-topics-deep-dive.md).
- [Package architecture](architecture/index.md).
- [Arrays, notebooks and external consumers](integration-guides-platforms.md).

## Operations and contribution

- [Performance measurement](performance-optimization-guide.md).
- [Troubleshooting](troubleshooting-faq.md).
- [Test commands and coverage](testing-guide.md), [test responsibilities](test-architecture-summary.md).
- [Contributor workflow](contributor-development-guide.md).
- [Capabilities and dated receipts](../STATUS.md), [backlog/history](../TODO.md).
- [Documentation maintenance](AGENTS.md).

## Conceptual applications and research

These guides discuss motivations or proposals, not additional installed APIs:

- [Research overview](research/index.md).
- [Case studies](case-studies-real-world-applications.md).
- [Interdisciplinary applications](interdisciplinary-applications-synergetics.md).
- [Learning theory](learning-theory-synergetics.md).
- [Philosophy and education](synergetics-philosophy-education.md).
- [Future research directions](future-implications-research-directions.md).

## Related

The separate [paper pipeline](../paper/) and
[manuscript status](manuscript/MANUSCRIPT_STATUS.md) describe manuscript work.
The root [README](../README.md) and [orchestrator](../run.py) describe package
entry points and their output side effects. Generated output is not current-run
evidence merely because a file exists.
