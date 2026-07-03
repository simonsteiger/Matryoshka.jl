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
        include("jet.jl")
        include("ad.jl")
    end
end
