"""
    NormalFamily <: Family

Normal (Gaussian) response family: `mu` is supplied by the predictor formula on
the identity scale; `sigma` is sampled with its own prior (default
`Exponential(1)`) unless targeted by a formula (distributional regression; not
yet supported in v0).

# Example
```julia
using Matryoshka

lik = @likelihood Normal y ~ x
lik.family isa NormalFamily
```
"""
struct NormalFamily <: Family end
to_family(::Type{Normal}) = NormalFamily()
parameters(::NormalFamily) = (:mu, :sigma)
links(::NormalFamily) = (mu = :identity, sigma = :log)
default_priors(::NormalFamily) = (sigma = Exponential(1),)

@model function normal_obs(eta, priors, y)
    sigma ~ priors.sigma
    y ~ product_distribution(Normal.(eta, sigma))  # identity link: mu == eta
    return nothing
end
obsmodel(::NormalFamily) = normal_obs
