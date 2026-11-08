# Computation maintenance: distinguish exact sequences from scores

Read [module purposes](README.md) and [canonical APIs](../../docs/api/index.md).
Keep `primorial(n)` prime-bound, not a product of the first n primes. Preserve
`scheherazade_power(n) = 1001**n`; base-1000 blocks cease to be separate Pascal
coefficients when carries occur. Audit decimal representation and numerator
handling for rational inputs instead of assuming every analysis uses a rational's
full mathematical value.

Preserve return dictionaries/dataclasses consumed by reporting and visualization.
Descriptive similarity/complexity scores are not mathematical proofs. Computation
functions do not all write output: explicit exporters/renderers and their callers
own destinations. Do not run demos against user artifacts. Tests live in
`../../tests/test_computation_*.py`; follow the active verification assignment.
