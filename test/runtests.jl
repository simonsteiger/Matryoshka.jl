using Matryoshka
using Test
using Aqua

@testset "Matryoshka.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(Matryoshka)
    end
    # Write your tests here.
end
