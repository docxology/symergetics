# Manuscript Style: Informative Structure and Source-Bound Claims

## Overview

Write an inspectable mathematical/software exposition. Distinguish derivations,
implemented behavior, historical receipts, proposed applications and empirical
results. A precise sentence with a source is preferable to “comprehensive,”
“revolutionary,” “universal,” “perfect” or “research-grade” without evidence.

## Markdown Formatting Standards

Use one H1 title in `00_title.md`, an H2 for each numbered section and descriptive
H3/H4 subsections. Preserve explicit anchors and update incoming heading links
when renaming. Use inline code for actual identifiers, not as a substitute for
mathematical definitions. Use descriptive link text without requiring decorative
symbols. Tables must identify units and comparison conventions.

## Image Integration Guidelines

Use source-relative image paths, concrete figure numbers and detailed captions.
For files in `paper/markdown/`, repository assets begin `../../output/`.
The [integration guide](INTEGRATION_GUIDE.md) defines exact resolution, caption
handling and aspect-preserving raster fitting. Missing/unsupported inputs fail;
SVG/PDF filename discovery does not provide a conversion stage.

## Figure Caption Standards

A caption states what the actual figure shows, input/scale where relevant,
quantity/units and interpretation limits. It must not assert that appearance
validates the mathematics or a real-world benefit. Example using an existing
source reference:

```markdown
![Figure 2: Quadray Representative (2,1,1,0)](../../output/geometric/coordinates/quadray_coordinate_2_1_1_0.png)

**Figure 2**: Coordinate example in the scale-1 rank-three embedding. The tuple (2,1,1,0) maps to XYZ (0,2,2); a common offset in all four slots leaves the position unchanged.
```

The source image may predate a source correction; do not claim fresh rendering
without a receipt. Keep long exact values in metadata when necessary rather
than overflowing a figure label. Label vertex means and mass centroids separately.

## Link and Reference Standards

Link actual `.py` implementations or canonical API docs instead of stale
extensionless GitHub paths and duplicated signatures. Citation metadata must
come from a real identifiable source; do not invent journal/arXiv status or
placeholder study names. Historical quotations need a locatable passage and
must be labeled as context, not empirical evidence. A DOI deposit and a current
development checkout have different revision boundaries.

## Mathematical Notation

- **Exact rational** means exact relative to specified rational input/algebra,
  not exact measurement, π, e or φ. A rational approximation is still approximate
  relative to its irrational target.
- **Quadray** means rank-three common-offset equivalence with four slots, not
  four independent Euclidean axes. Minimum-zero is not zero-sum normalization.
- **Lattice** distinguishes the full BCC integer image from the selected FCC/IVM
  neighbor graph. State the embedding scale and snapping policy.
- **Volume** distinguishes native `abs(det)/4` units from Cartesian cubic units;
  native volume can be fractional. Default solid ratios require declared sizes.
- **Centers** distinguish exact vertex mean from uniform-solid mass centroid.
- **Primorial** states the convention: implemented n# multiplies primes ≤ n;
  13# = 30,030, while the product of the first six primes uses a different index.
- **Digit patterns** state base, search interval and the no-carry condition for
  binomial coefficients read as base-1000 blocks.

[Foundations](markdown/03_mathematical_foundations.md) is the canonical manuscript
mathematical account; [research boundaries](../docs/research/index.md) gives
source/model limits.

## Code Block Standards

Runnable snippets use real APIs with complete imports, supported inputs and no
omitted implementation. Do not include `pass`, fake defaults, undefined helper
calls or a speculative domain class as software documentation. Prefer links to
[API contracts](../docs/api/index.md) and actual examples to another drifting API
copy. If discussing a proposed algorithm, use clearly labeled prose with its
inputs, assumptions and acceptance evidence instead of a runnable-looking stub.
Keep displayed snippets short; the renderer truncates long code blocks.

## Section Structure

Use the twelve numbered modules linked from [paper structure](README.md#paper-structure).
A section should state its question, assumptions, method/source, result/status
and limitations. “Results” must not silently mix known identities, source review,
test execution and empirical efficacy. Cross-link prerequisites rather than
repeating incompatible mathematical tables across sections.

## Quality Assurance

Before publication, the integration owner should check structural validation,
actual reference targets and a freshly rendered PDF in isolation. Review
legibility, aspect, units, caption numbering and supported image formats.
These checks do not validate scientific claims. Historical coverage/test counts
remain dated, revision-bound receipts; see [test evidence](TEST_EVIDENCE.md).
No visual or documentation edit alone proves current green tests.

## Common Mistakes to Avoid

Do not claim all operations are exact, all solids are implemented, general XYZ
inversion is lossless, exact arithmetic implies physical accuracy, or named
geometric constants are independent measurements. Remove unsupported percentages
and success stories rather than giving them invented sources. Do not describe
`generate_paper.py` as PDF-only; use the [direct-render workflow](QUICK_START.md).

## Style Guide Updates

When terminology or headings change, update affected source modules and backlinks
without rewriting immutable publications or generated artifacts. Match evidence
to the reviewed revision and state what was actually exercised.
