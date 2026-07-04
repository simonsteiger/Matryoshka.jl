# Matryoshka.jl

brms for Turing.jl — formula-based Bayesian regression compiled to composable Turing submodels.

[![Build Status](https://github.com/simonsteiger/Matryoshka.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/simonsteiger/Matryoshka.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/simonsteiger/Matryoshka.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/simonsteiger/Matryoshka.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)<a href="https://simonsteiger.github.io/Matryoshka.jl/dev"><img src="/docs/src/assets/logo.svg" align="right" alt="Matryoshka logo" style="height: 180px;"></a>

Matryoshka is a thin front-end, not a modeling engine of its own. 

The macros `@likelihood` and `@priors` parse a formula and a prior spec; `model(likelihood, priors, data)` lowers them into nested `DynamicPPL` submodels.

Everything downstream is for the inference backend to take care of (tested only on Turing!).

## Example

The example below shows how to use Matryoshka to fit a single categorical predictor to a continuous response variable:

```julia
using Matryoshka, Turing, FlexiChains
using PalmerPenguins, DataFrames, CategoricalArrays
using StatsBase: denserank

# We will z-standardise the outcome
standardise(x) = (x .- mean(x)) ./ std(x)

data = DataFrame(PalmerPenguins.load())

# There's one poor penguin with missings
dropmissing!(data)
transform!(data, :bill_length_mm => standardise => identity)
transform!(data, :species => categorical => identity)

# Bill length by species, Normal family
likelihood = @likelihood Normal bill_length_mm ~ species

# Standard Normal priors on intercept and species coefs
priors = @priors begin
    intercept ~ Normal(0, 1)
    b ~ Normal(0, 1)
end

# Now we create a `DynamicPPL.Model`, then Turing takes over
bill_model = model(likelihood, priors, data)
chain = sample(bill_model, NUTS(), 1000)

# Spoiler alert: Adelie have shortest bills (they're cute!)
summarystats(chain)
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
