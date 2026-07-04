using Matryoshka
using Distributions, DynamicPPL, Turing, Test
using Logging: Logging

df = (
    y = [1.1, 2.3, 0.9, 1.8, 2.5, 1.2],
    x = [0.5, 1.0, 0.2, 0.9, 1.4, 0.3],
    g = ["a", "a", "b", "b", "c", "c"],
)

@testset "model() returns a plain DynamicPPL.Model" begin
    lik = @likelihood Normal begin
        y ~ x + (1 | g)
    end
    pri = @priors begin
        intercept ~ Normal(0, 10)
        b ~ Normal(0, 1)
        sd ~ Exponential(1)
        sigma ~ Exponential(1)
    end
    m = model(lik, pri, df)
    @test m isa DynamicPPL.Model
    @test m.args.recipe isa Matryoshka.Recipe
    vns = string.(keys(DynamicPPL.VarInfo(m)))
    @test "intercept" in vns
    @test "sigma" in vns
    @test any(startswith("g."), vns)      # g.sd, g.z
    @test any(startswith("b"), vns)
end

@testset "logjoint equality vs handwritten Turing model" begin
    lik = @likelihood Normal begin
        y ~ x
    end
    pri = @priors begin
        intercept ~ Normal(0, 10)
        b ~ Normal(0, 1)
        sigma ~ Exponential(1)
    end
    m = model(lik, pri, df)

    @model function hand(x, y)
        intercept ~ Normal(0, 10)
        b ~ arraydist([Normal(0, 1)])
        sigma ~ Exponential(1)
        mu = intercept .+ x .* b[1]
        y ~ product_distribution(Normal.(mu, sigma))
    end
    for θ in (
            (intercept = 1.0, b = [0.5], sigma = 1.0),
            (intercept = 0.0, b = [-1.2], sigma = 0.3),
        )
        @test logjoint(m, θ) ≈ logjoint(hand(collect(df.x), collect(df.y)), θ)
    end
end

@testset "poisson + bernoulli links" begin
    dfp = (y = [0, 2, 5, 1], x = [0.1, 0.5, 1.5, 0.2])
    lik = @likelihood Poisson y ~ x
    pri = @priors begin
        intercept ~ Normal(0, 5)
        b ~ Normal(0, 1)
    end
    m = model(lik, pri, dfp)
    @model function handp(x, y)
        intercept ~ Normal(0, 5)
        b ~ arraydist([Normal(0, 1)])
        y ~ product_distribution(Poisson.(exp.(intercept .+ x .* b[1])))
    end
    θ = (intercept = 0.5, b = [0.8])
    @test logjoint(m, θ) ≈ logjoint(handp(collect(dfp.x), collect(dfp.y)), θ)
end

@testset "sampling smoke test" begin
    lik = @likelihood Normal y ~ x + (1 | g)
    pri = @priors b ~ Normal(0, 1)      # everything else: defaults
    m = model(lik, pri, df)
    chain = Logging.with_logger(Logging.NullLogger()) do
        sample(m, NUTS(), 200; progress = false)
    end
    # Turing >= 0.45 returns FlexiChains, not MCMCChains; reflect on the chain's own
    # module rather than assuming MCMCChains internals (see test/spike_notes.md).
    FC = parentmodule(typeof(chain))
    @test FC.niters(chain) == 200
    pnames = string.(FC.parameters(chain))
    @test "intercept" in pnames
    @test "sigma" in pnames
    @test any(startswith("g."), pnames)   # g.sd, g.z
    @test any(startswith("b"), pnames)
end
