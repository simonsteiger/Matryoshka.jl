struct BernoulliFamily <: Family end
to_family(::Type{Bernoulli}) = BernoulliFamily()
parameters(::BernoulliFamily) = (:mu,)
links(::BernoulliFamily) = (mu = :logit,)
default_priors(::BernoulliFamily) = NamedTuple()

# obsmodels receive the link-scale linear predictor eta. BernoulliLogit works
# on the log-odds scale directly, so extreme eta never saturates to p == 1.0
# (which would trip Bernoulli's check_args under ForwardDiff Duals).
@model function bernoulli_obs(eta, priors, y)
    y ~ product_distribution(BernoulliLogit.(eta))
    return nothing
end
obsmodel(::BernoulliFamily) = bernoulli_obs
