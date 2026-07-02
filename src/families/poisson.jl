struct PoissonFamily <: Family end
to_family(::Type{Poisson}) = PoissonFamily()
parameters(::PoissonFamily) = (:mu,)
links(::PoissonFamily) = (mu = :log,)
default_priors(::PoissonFamily) = NamedTuple()

@model function poisson_obs(mu, priors, y)
    y ~ product_distribution(Poisson.(mu))
    return nothing
end
obsmodel(::PoissonFamily) = poisson_obs
