using Matryoshka
using Test

const GROUP = get(ENV, "GROUP", "All")

@testset "Matryoshka.jl" begin
    if GROUP in ("All", "Core")
        include("families.jl")
        include("priors.jl")
        include("likelihood.jl")
        include("components.jl")
        include("model.jl")
        include("introspection.jl")
        include("errors.jl")
        include("predict.jl")
    end
    if GROUP in ("All", "Recovery")
        include("recovery.jl")
    end
    if GROUP in ("All", "Static")
        include("aqua.jl")
        isfile(joinpath(@__DIR__, "jet.jl")) && include("jet.jl")   # Task 12
        isfile(joinpath(@__DIR__, "ad.jl")) && include("ad.jl")     # Task 12
    end
end
