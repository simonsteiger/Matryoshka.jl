struct BernoulliFamily <: Family end
to_family(::Type{Bernoulli}) = BernoulliFamily()
parameters(::BernoulliFamily) = (:mu,)
links(::BernoulliFamily) = (mu = :logit,)
default_priors(::BernoulliFamily) = NamedTuple()

@model function bernoulli_obs(mu, priors, y)
    # check_args=false: mu can saturate to exactly 0.0/1.0 in Float64 for extreme
    # linear predictors reached during NUTS step-size search; these are valid
    # boundary probabilities and must not throw DomainError mid-sampling.
    y ~ product_distribution(Bernoulli.(mu; check_args = false))
    return nothing
end
obsmodel(::BernoulliFamily) = bernoulli_obs
