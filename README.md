# Matryoshka.jl

brms for Turing.jl — formula-based Bayesian regression compiled to composable Turing submodels.

[![Build Status](https://github.com/simonsteiger/Matryoshka.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/simonsteiger/Matryoshka.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/simonsteiger/Matryoshka.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/simonsteiger/Matryoshka.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)<a href="https://simonsteiger.github.io/Matryoshka.jl/dev"><img src="/docs/src/assets/logo.svg" align="right" alt="Matryoshka logo" style="height: 180px;"></a>

Matryoshka is a thin front-end, not a modeling engine of its own. 

The macros `@likelihood` and `@priors` parse a formula and a prior spec; `model(likelihood, priors, data)` lowers them into nested `DynamicPPL` submodels and hands back a **plain `DynamicPPL.Model`**.

Everything downstream is for the inference backend to take care of (this very early version only tests Turing!).

## Quickstart

```julia
using Matryoshka, Distributions, Turing

# Can also be written without begin / end
likelihood = @likelihood Normal begin
    y ~ x + (1 | g)
end

priors = @priors begin
    intercept ~ TDist(3)
    b ~ Normal(0, 1)
    g.sd ~ Exponential(0.5)
end

m = model(likelihood, priors, data) # just a DynamicPPL.Model
chain = sample(m, NUTS(), 1000) # Turing or otherwise take over
m_new = model(m, newdata)
predict(m_new, chain) # posterior predictive check
```

Lots of models types are still unsupported in this v0 draft!

## Parameter naming per @priors

Chain parameter names equal the `@priors` targets you write, with one exception — population-level slopes are one vector-valued `VarName` `b`, not per-name `VarName`s.

| prior target (`@priors`)  | chain name                    |
|:--------------------------|:------------------------------|
| `intercept`               | `intercept`                   |
| `b` (class, all slopes)   | `b`                           |
| `g.sd` (exact, group `g`) | `g.sd`                        |
| `sigma`                   | `sigma`                       |

## Not supported yet

- Distributional regression
- Multivariate response models
- GP / smooth terms
