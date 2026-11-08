# Example maintenance: executable workflows and safe destinations

Use [the script map](README.md) and [canonical APIs](../docs/api/index.md).
Examples are standalone scripts, but dependencies and output behavior vary.
Select Python 3.13 explicitly and install scientific extras for renderer demos.
Do not treat demos or `run.py --examples-only` as read-only checks: figures,
logs and reports can mutate repository output. Use an actual supported explicit
external destination; an ignored `output_dir` keyword is not isolation.

Keep exactness, coordinate gauges and native/XYZ units visible in explanations.
Do not infer successful execution from a printed success message: several demos
catch exceptions and return status/results. Poster section selection describes
unreleased main only. The interoperability script uses real sibling packages,
Node.js and C++17; do not replace unavailable consumers with fabricated data.
Verification follows the active assignment; never run every demo automatically.
