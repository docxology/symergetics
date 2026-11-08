# Utility maintenance: preserve conversion and report schemas

Read [utility contracts](README.md) and [API reference](../../docs/api/index.md).
`conversion.py`, `mnemonics.py` and `reporting.py` support computation, renderers
and demos. Preserve report input/output schemas and explicit export destinations.

Do not describe denominator-limited float conversion as recovery of an exact
intended decimal. Keep exact string parsing, binary floats, nearest-lattice
projection and display formatting distinct. Mnemonic associations are explanatory
text, not validated learning outcomes. Use actual analysis dictionaries when
exercising summaries; do not pass numeric objects where dictionaries are expected.
Tests are top-level `../../tests/test_utils_*.py`; the active assignment controls
whether verification is permitted.
