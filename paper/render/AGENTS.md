# Renderer Work: Presentation and Artifact Boundaries

Read [README.md](README.md) and [figure integration](../INTEGRATION_GUIDE.md).
The renderer consumes the twelve numbered sources and existing figures;
it does not establish mathematical correctness or regenerate all examples.
Use explicit source/output paths in an isolated checkout when rendering is
assigned. Mermaid may additionally write under `paper/mermaid_images/`.
Preserve generated/user-owned files and immutable publication outputs.
Referenced images resolve exactly relative to `paper/markdown/`; never re-add
basename guesses or image placeholders. Preserve raster aspect and authored
caption numbers, and test repeated occurrences rather than suppressing them.

Embed the bundled Unicode fonts; never strip mathematical operators or units
to make text render. Keep table columns within the page frame and preserve empty
cells. Check real PDF text and placement, not style names or documentation wording.
