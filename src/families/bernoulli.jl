"""
    BernoulliFamily <: Family

Bernoulli response family for binary outcomes: `mu` (success probability) is
supplied by the predictor formula on the logit scale. `obsmodel` applies
`BernoulliLogit` directly to the link-scale linear predictor rather than
inverse-linking to a probability first, so extreme `eta` never saturates
`p == 1.0` under autodiff.

# Example
```julia
using Matryoshka

lik = @likelihood Bernoulli y ~ x
lik.family isa BernoulliFamily
```
"""
struct BernoulliFamily <: Family end
to_family(::Type{Bernoulli}) = BernoulliFamily()
parameters(::BernoulliFamily) = (:mu,)
links(::BernoulliFamily) = (mu = :logit,)
default_priors(::BernoulliFamily) = NamedTuple()

# obsmodels receive the link-scale linear predictor eta. BernoulliLogit works
# on the log-odds scale directly, so extreme eta never saturates to p == 1.0
# (which would trip Bernoulli's check_args under ForwardDiff Duals).
@model function bernoulli_obs(eta, priors, y)
    y ~ withdims(product_distribution(BernoulliLogit.(eta)), Dim{:obs}(1:length(eta)))
    return nothing
end
obsmodel(::BernoulliFamily) = bernoulli_obs
