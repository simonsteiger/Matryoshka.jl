"""
    PoissonFamily <: Family

Poisson response family for count outcomes: `mu` (rate) is supplied by the
predictor formula on the log scale.

# Example
```julia
using Matryoshka

lik = @likelihood Poisson y ~ x
lik.family isa PoissonFamily
```
"""
struct PoissonFamily <: Family end
to_family(::Type{Poisson}) = PoissonFamily()
parameters(::PoissonFamily) = (:mu,)
links(::PoissonFamily) = (mu = :log,)
default_priors(::PoissonFamily) = NamedTuple()

@model function poisson_obs(eta, priors, y)
    y ~ product_distribution(Poisson.(exp.(eta)))
    return nothing
end
obsmodel(::PoissonFamily) = poisson_obs
