```@raw html
---
layout: home

hero:
  name: "Matryoshka.jl"
  tagline: A brms-like interface to probabilistic modelling
  image:
    src: /logo.svg
    alt: Matryoshka.jl
  actions:
    - theme: brand
      text: API Reference
      link: /api
    - theme: alt
      text: View on GitHub
      link: https://github.com/simonsteiger/Matryoshka.jl
---
```

# Matryoshka.jl

Matryoshka.jl builds Bayesian regression models from composable components and returns a `DynamicPPL.Model`.

## Installation

This package is not registered with the General registry.
You can install it via URL:

```julia
import Pkg
Pkg.add(url = "https://github.com/simonsteiger/Matryoshka.jl")
```

## Get started

The example below shows how to use Matryoshka to fit a single categorical predictor to a continuous response variable:

```@example index
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

# intercept = Adelie
# b[1] = Chinstrap
# b[2] = Gentoo
# Adelie have shortest bills (they're cute!)
summarystats(chain)
```
