using Nutshell
using Test
using Aqua

@testset "Nutshell.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(Nutshell)
    end
    # Write your tests here.
end
