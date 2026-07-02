struct NormalFamily <: Family end
to_family(::Type{Normal}) = NormalFamily()
parameters(::NormalFamily) = (:mu, :sigma)
links(::NormalFamily) = (mu = :identity, sigma = :log)
default_priors(::NormalFamily) = (sigma = Exponential(1),)

@model function normal_obs(mu, priors, y)
    sigma ~ priors.sigma
    y ~ product_distribution(Normal.(mu, sigma))
    return nothing
end
obsmodel(::NormalFamily) = normal_obs
