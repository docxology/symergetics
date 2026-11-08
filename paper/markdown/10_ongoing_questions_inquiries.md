## Ongoing Questions and Inquiries

These are unanswered research questions, not promised features or demonstrated
advantages. The current implementation and its domains are described in the
[API reference](../../docs/api/index.md). The
[research methodology guide](../../docs/research/index.md#research-methodology)
defines evidence requirements.

### Open Research Questions

- **Representation tradeoffs:** For a specified geometric workload, how do
  quadray and exact Cartesian representations compare in operation count,
  intermediate bit length and implementation complexity? Both can be exact.
- **Quantization:** How does nearest-integer quadray snapping affect a named
  continuous dataset at a declared embedding scale? Compare error against
  observational uncertainty rather than assuming exact preservation.
- **Approximation:** Which downstream results are sensitive to rational
  approximations of irrational roots/constants, and what error bounds can be
  established independently?
- **Surface domain:** How should additional geometric operations specify valid
  topology and intersection assumptions? Current volume/moment routines already
  support declared fan surfaces beyond default solids; they do not promise
  detection of every invalid surface.

### Computational Challenges

- Measure time and memory against numerator/denominator bit length, sequence
  length and search scope. Exact rational arithmetic removes rational rounding,
  not combinatorial cost or resource limits.
- Compare a fixed task with float, arbitrary-precision and symbolic alternatives
  under matched error requirements; do not conflate speed with accuracy.
- Specify serialization of rational values and coordinate gauges so that
  float-only consumers do not silently lose information.
- Evaluate parallelization only for an identified workload. A modular package
  does not itself constitute a distributed runtime or hardware accelerator.

### Research Applications

A useful proposal identifies the model component, domain data, null comparator
and observation that could refute it. Cognitive, material, biological, climate,
quantum and financial claims need domain evidence; no universal advantage
follows from rational storage. See [proposed applications](09_research_applications.md).

### Technical Development

Potential investigations include explicit approximation-error reporting,
representation-aware pattern statistics and accessible figure design. A finite
search can suggest a conjecture but does not prove it for all inputs. Benchmark
and correctness statements must name the implemented function and domain.

### Community and Collaboration

Share source revision, environment, exact inputs, units, assumptions and output
provenance. Separate independently checked derivations from source review,
software tests and empirical replication. Preserve failed comparisons and null
results rather than selecting only visually striking patterns.

### Future Directions

A higher-dimensional generalization would require a newly defined representation
and metric; the current four quadray slots do not already implement four physical
dimensions. Domain integrations should only become API documentation after real
interfaces and consumer examples exist. See the
[research agenda](../../docs/future-implications-research-directions.md).

### Predictive Histories and Futures

Historical publications and generated PDFs describe their own source versions.
A development-checkout correction does not revise an immutable deposit or prove
future impact. Continued research is optional investigation, not evidence that
all proposed applications will succeed.
