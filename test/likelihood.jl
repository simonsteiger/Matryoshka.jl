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

@testset "@likelihood rejects non-tilde lines" begin
    @test_throws LoadError @eval @likelihood Normal begin
        y = x
    end
end

@testset "@likelihood rejects empty block" begin
    @test_throws LoadError @eval @likelihood Normal begin end
end

@testset "@likelihood macro hygiene: no StatsModels needed in caller scope" begin
    # `formula_line` builds `StatsModels.@formula(...)` for the escaped formula; it
    # must resolve via `GlobalRef`, not rely on `StatsModels` being bound in the
    # caller's module (which only `using Matryoshka` does not provide).
    m = Module(:LikelihoodHygieneFresh)
    Core.eval(m, :(using Matryoshka, Distributions))
    lik = Core.eval(m, :(@likelihood Normal y ~ x + (1 | g)))
    @test lik isa Likelihood
    @test lik.family isa NormalFamily
end

@testset "@priors macro hygiene: no StatsModels needed in caller scope" begin
    m = Module(:PriorsHygieneFresh)
    Core.eval(m, :(using Matryoshka, Distributions))
    pri = Core.eval(m, :(@priors intercept ~ Normal(0, 10)))
    @test pri isa Matryoshka.Priors
end
