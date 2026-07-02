module Matryoshka

using DynamicPPL: DynamicPPL, @model, to_submodel
using Distributions
using StatsModels: StatsModels
using Tables: Tables

include("families/interface.jl")
include("families/normal.jl")
include("families/bernoulli.jl")
include("families/poisson.jl")
include("priors.jl")

export Family, NormalFamily, BernoulliFamily, PoissonFamily
export parameters, links, default_priors, obsmodel
export @priors, Priors

end
