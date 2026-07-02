using Matryoshka
using Matryoshka: Likelihood, response_formula, extra_formulas, NormalFamily, PoissonFamily
using StatsModels, Test

@testset "@likelihood" begin
    lik = @likelihood Normal begin
        y ~ x + (1 | g)
    end
    @test lik isa Likelihood
    @test lik.family === NormalFamily()
    f = response_formula(lik)
    @test f isa StatsModels.FormulaTerm
    @test StatsModels.termvars(f) == [:y, :x, :g]
    @test extra_formulas(lik) === ()

    lik2 = @likelihood Poisson y ~ x
    @test lik2.family === PoissonFamily()

    lik3 = @likelihood Normal begin
        y ~ x
        sigma ~ z
    end
    @test length(lik3.formulas) == 2
end
