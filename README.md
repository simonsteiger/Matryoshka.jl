# Matryoshka.jl

brms for Turing.jl — formula-based Bayesian regression compiled to composable Turing submodels.

[![Build Status](https://github.com/simonsteiger/Matryoshka.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/simonsteiger/Matryoshka.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/simonsteiger/Matryoshka.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/simonsteiger/Matryoshka.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Matryoshka is a thin front-end, not a modeling engine of its own. `@likelihood`
and `@priors` parse a formula and a prior spec; `model(lik, pri, tbl)` lowers
them into nested `DynamicPPL` submodels and hands back a **plain
`DynamicPPL.Model`**. Everything downstream — `sample`, `predict`, chain
objects — is Turing's own, untouched. There is no wrapper struct to learn: if
you know Turing, you already know how to use the output.

## Quickstart

```julia
using Matryoshka, Distributions, Turing

lik = @likelihood Normal begin
    y ~ x + (1 | g)
end

pri = @priors begin
    intercept ~ TDist(3)
    b ~ Normal(0, 1)
    g.sd ~ Exponential(0.5)
end

m = model(lik, pri, df)             # plain DynamicPPL.Model, recipe stored in args
chain = sample(m, NUTS(), 1000)     # plain Turing; FlexiChains by default (Turing ≥ 0.45)

m_new = model(m, newdata)           # rebuild with training schema, y missing
predict(m_new, chain)               # plain Turing
```

(v0 does not yet support distributional regression — a formula on `sigma`
such as `sigma ~ x` — so the spec's `sigma ~ z` example line is left out
above; see "Not yet" below. A `@priors` line like `sigma ~ Exponential(...)`
is unrelated and always fine: it assigns a *prior*, not a formula.)

## Naming: prior target = chain name

Chain parameter names equal the `@priors` targets you write, with one
exception — population-level slopes are one vector-valued `VarName` `b`, not
per-name `VarName`s (`DynamicPPL` does not support per-key `VarName`s inside a
`Dict`; see `test/spike_notes.md` Q3).

| prior target (`@priors`) | chain name                    |
|:-------------------------|:-------------------------------|
| `intercept`               | `intercept`                    |
| `b` (class, all slopes)   | `b` (single vector; pull coefficient `i` across draws with `getindex.(chain[:b], i)` — positional, not `b.x`/`b[1]` as a real VarName) |
| `g.sd` (exact, group `g`) | `g.sd`                         |
| `sigma`                   | `sigma`                        |

## Not yet

- Distributional regression (formulas on non-mean parameters, e.g. `sigma ~ x`)
- Multivariate response models
- GP / smooth terms
- Prediction for new factor levels (`allow_new_levels`) — `rebuild` raises an
  `ArgumentError` on an unseen grouping-variable level
