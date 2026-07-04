using Matryoshka, Distributions, Test

df = (y = [1.0, 2.0, 3.0], x = [0.1, 0.2, 0.3], g = ["a", "b", "a"])

@testset "default_priors table" begin
    lik = @likelihood Normal y ~ x + (1 | g)
    tab = default_priors(lik, df)
    targets = [r.target for r in tab]
    @test "intercept" in targets
    @test "b.x" in targets
    @test "g.sd" in targets
    @test "sigma" in targets
    row = tab[findfirst(==("g.sd"), targets)]
    @test row.class == "sd"
    @test row.prior == Exponential(1)
end
