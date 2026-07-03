using Matryoshka
using Test
using Aqua

@testset "Matryoshka.jl" begin
    @testset "Aqua" begin
        Aqua.test_all(Matryoshka; stale_deps = (ignore = [:Turing, :StatsModels, :Tables],))
    end
    include("families.jl")
    include("priors.jl")
    include("likelihood.jl")
    include("components.jl")
    include("model.jl")
    include("introspection.jl")
    include("errors.jl")
end
