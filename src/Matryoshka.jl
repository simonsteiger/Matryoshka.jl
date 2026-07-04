module Matryoshka

using DynamicPPL: DynamicPPL, @model, to_submodel, filldist, arraydist
using Distributions
using StatsModels: StatsModels
using Tables: Tables

include("families/interface.jl")
include("families/normal.jl")
include("families/bernoulli.jl")
include("families/poisson.jl")
include("priors.jl")
include("likelihood.jl")
include("components/interface.jl")
include("components/intercept.jl")
include("components/fixedeffects.jl")
include("components/randomintercept.jl")
include("lowering.jl")
include("model.jl")

export Family, NormalFamily, BernoulliFamily, PoissonFamily
export parameters, links, default_priors, obsmodel
export @priors, Priors
export @likelihood, Likelihood
export AbstractComponent
export compprefix, submodel, priorslots, rebuild
export model

end
