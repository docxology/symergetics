# Symergetics Paper: Source Modules and Publication Boundaries

## Overview

The editable manuscript describes selected rational-arithmetic, quadray geometry,
volume and number-pattern computations. It is a software/mathematical exposition,
not evidence of universal precision, comparative speed or empirical success in
physics, biology, finance, cognition or education. Proposed applications are
labeled separately from implemented interfaces.

These pages describe the development checkout. Historical PDFs, DOI deposits
and immutable releases describe their own versions; editing sources does not
update a deposited paper. The title page's 2025 date/version are historical
metadata, not a package version declaration.

## Author Information

Daniel Ari Friedman · daniel@activeinference.institute  
ORCID: [0000-0001-6232-9096](https://orcid.org/0000-0001-6232-9096)  
Affiliation recorded in the source: Active Inference Institute.

## Paper Structure

The renderer loads twelve numbered source sections in order:

1. [Title and historical metadata](markdown/00_title.md)
2. [Abstract](markdown/01_abstract.md)
3. [Introduction and scope](markdown/02_introduction.md)
4. [Mathematical foundations](markdown/03_mathematical_foundations.md)
5. [Implementation architecture](markdown/04_system_architecture.md)
6. [Computational methods](markdown/05_computational_methods.md)
7. [Results and evidence limits](markdown/06_results.md)
8. [Geometric applications](markdown/07_geometric_applications.md)
9. [Number-pattern analysis](markdown/08_pattern_discovery.md)
10. [Proposed research applications](markdown/09_research_applications.md)
11. [Open questions](markdown/10_ongoing_questions_inquiries.md)
12. [Conclusion and declarations](markdown/11_conclusion.md)

## Quick Start

Use the [Python 3.13 setup and direct-render commands](QUICK_START.md).
`generate_paper.py` is a broader orchestration script that can clear outputs,
run tests and regenerate examples; it is **not** a PDF-only command. Work in an
isolated checkout when generating, and do not overwrite user artifacts or
immutable publication outputs.

## Mathematical and Evidence Contracts

- [Foundations](markdown/03_mathematical_foundations.md): rational/irrational
  boundaries, rank-three quadray gauge, lattice distinction and native units.
- [API reference](../docs/api/index.md): actual callable interfaces.
- [Research guide](../docs/research/index.md): implemented components versus
  model assumptions, proposals and empirical evidence.
- [Test evidence](TEST_EVIDENCE.md): historical source inventories and execution
  receipts, not current suite counts or universal correctness claims.

## Figure Resolution and Readable PDFs

The renderer resolves referenced figures; it does not automatically turn every
output file into a verified finding. See the [integration guide](INTEGRATION_GUIDE.md)
for exact source-relative lookup, aspect-preserving raster fitting and artifact provenance.
Use [style guidance](STYLE_GUIDE.md) for informative titles, units and bounded
captions. A generated validation report is a dated artifact, not current proof.

The [renderer contract](render/README.md#pdf-heading-pagination-unicode-and-pipe-tables)
describes H1–H4 heading/body pagination, Unicode fonts, literal inline code and
bounded pipe tables. Review an actual PDF for heading orphans and column
placement; supported Markdown is deliberately narrower than a browser's layout
engine or a scholarly LaTeX publishing system.

## Dependencies

The package's [pyproject.toml](../pyproject.toml) defines supported Python and
optional dependencies. Scientific examples here use Python 3.13; leave the
repository interpreter pin unchanged. Rendering dependencies are listed in
[render/requirements.txt](render/requirements.txt). No PyPI `symergetics`
installation is advertised: use a checkout or an actual release wheel through
the [root installation guide](../README.md).

## Contributing

Edit numbered sources rather than generated PDFs. Keep API details linked to
the implementation reference. Any new scientific claim needs identifiable
sources, assumptions and evidence appropriate to its type. Validate structure,
references and rendered readability in a disposable output location before
publication; those checks do not replace mathematical or empirical review.

## Citation

Use the metadata at [paper DOI 10.5281/zenodo.17114390](https://doi.org/10.5281/zenodo.17114390)
for the deposited paper and [repository DOI 10.5281/zenodo.17114383](https://doi.org/10.5281/zenodo.17114383)
for the archived software as applicable. State the checkout revision when citing
development behavior. No arXiv publication or journal status is asserted here.

## License

The repository [LICENSE](../LICENSE) is Apache 2.0, not MIT. Refer to deposited
metadata for any separately distributed publication terms.

[Documentation map](../docs/index.md) · [Manuscript status](../docs/manuscript/MANUSCRIPT_STATUS.md)
