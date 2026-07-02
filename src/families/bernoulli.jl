struct BernoulliFamily <: Family end
to_family(::Type{Bernoulli}) = BernoulliFamily()
parameters(::BernoulliFamily) = (:mu,)
links(::BernoulliFamily) = (mu = :logit,)
default_priors(::BernoulliFamily) = NamedTuple()

@model function bernoulli_obs(mu, priors, y)
    y ~ product_distribution(Bernoulli.(mu))
    return nothing
end
obsmodel(::BernoulliFamily) = bernoulli_obs
