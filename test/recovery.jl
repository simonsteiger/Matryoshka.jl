using Matryoshka
using Distributions, DynamicPPL, Turing, StableRNGs, Statistics, Test
using Logging: Logging
using DimensionalData: DimensionalData, At
include("testutils.jl")
using .MatryoshkaTestUtils: check_numerical, resolve_param, two_sample_ks

rng = StableRNG(468)

@testset "normal recovery + conjugate check via condition" begin
    n = 100
    x = randn(rng, n)
    y = 1.0 .+ 0.7 .* x .+ 0.5 .* randn(rng, n)
    lik = @likelihood Normal y ~ x
    m = model(lik, @priors(intercept ~ Normal(0, 10)), (; x, y))
    chain = Logging.with_logger(Logging.NullLogger()) do
        sample(rng, m, NUTS(), 1000; progress = false)
    end
    check_numerical(chain, ["intercept" => 1.0, "b[1]" => 0.7, "sigma" => 0.5]; atol = 0.2)

    # transparency payoff: condition sigma with vanilla DynamicPPL → known-variance
    # conjugate posterior for intercept-only model
    y0 = randn(rng, 100) .+ 2.0
    m0 = model((@likelihood Normal y ~ 1), (@priors intercept ~ Normal(0, 10)), (; y = y0))
    mc = DynamicPPL.condition(m0, sigma = 1.0)
    ch0 = Logging.with_logger(Logging.NullLogger()) do
        sample(rng, mc, NUTS(), 2000; progress = false)
    end
    prior_var = 100.0                        # Normal(0,10) variance
    post_var = 1 / (1 / prior_var + length(y0) / 1.0)
    post_mean = post_var * sum(y0) / 1.0
    check_numerical(ch0, ["intercept" => post_mean]; atol = 0.2)
    @test isapprox(var(vec(resolve_param(ch0, "intercept"))), post_var; rtol = 0.3)
end

@testset "poisson recovery" begin
    n = 100
    x = randn(rng, n)
    y = rand.(rng, Poisson.(exp.(0.2 .+ 0.5 .* x)))
    m = model((@likelihood Poisson y ~ x), @priors(b ~ Normal(0, 1)), (; x, y))
    chain = Logging.with_logger(Logging.NullLogger()) do
        sample(rng, m, NUTS(), 1000; progress = false)
    end
    check_numerical(chain, ["intercept" => 0.2, "b[1]" => 0.5]; atol = 0.2)
end

@testset "bernoulli recovery" begin
    n = 300
    x = randn(rng, n)
    p = @. inv(1 + exp(-(0.3 + 1.0 * x)))
    y = rand.(rng, Bernoulli.(p))
    m = model((@likelihood Bernoulli y ~ x), @priors(b ~ Normal(0, 1)), (; x, y))
    chain = Logging.with_logger(Logging.NullLogger()) do
        sample(rng, m, NUTS(), 1000; progress = false)
    end
    check_numerical(chain, ["intercept" => 0.3, "b[1]" => 1.0]; atol = 0.2)
end

@testset "random intercept recovery + KS predict vs handwritten" begin
    rng2 = StableRNG(42)
    ngroups, nper = 30, 10
    g = repeat(string.(1:ngroups); inner = nper)
    u = 0.8 .* randn(rng2, ngroups)
    x = randn(rng2, ngroups * nper)
    y = 1.0 .+ 0.5 .* x .+ u[parse.(Int, g)] .+ 0.3 .* randn(rng2, ngroups * nper)
    m = model((@likelihood Normal y ~ x + (1 | g)), @priors(b ~ Normal(0, 1)), (; x, y, g))
    chain = Logging.with_logger(Logging.NullLogger()) do
        sample(rng2, m, NUTS(), 1000; progress = false)
    end
    # 30 groups is too few to recover the population sd 0.8; the model recovers the
    # realized spread of the drawn group effects, so target std(u), not 0.8.
    check_numerical(
        chain,
        ["intercept" => 1.0, "b[1]" => 0.5, "g.sd" => std(u), "sigma" => 0.3];
        atol = 0.2,
    )

    # predictions from our model vs handwritten model, same chain params: KS
    m_new = model(m, (x = x[1:50], g = g[1:50]))
    preds = predict(StableRNG(7), m_new, chain)
    # extract predicted y draws for obs 1 (y is a single vector-valued VarName;
    # each draw is a length-50 vector, one entry per newdata row — see
    # test/predict.jl) and compare against normal draws implied by chain means
    # (coarse distributional check)
    ydraws = vec(getindex.(preds[@varname(y)], 1))
    # obs 1 belongs to group g[1] == "1"; unique(g)-first-appearance order puts
    # that group at position 1 in the "g.z" vector (see src/lowering.jl) — add
    # its non-centered random-intercept contribution sd*z[1] (per-draw product,
    # not mean(sd)*mean(z)) so μ̂ is apples-to-apples with predict()'s ydraws.
    z1 = getindex.(resolve_param(chain, "g.z"), 1)
    sd_g = resolve_param(chain, "g.sd")
    μ̂ = mean(resolve_param(chain, "intercept")) + x[1] * mean(resolve_param(chain, "b[1]")) +
        mean(sd_g .* z1)
    ref = μ̂ .+ 0.35 .* randn(StableRNG(8), length(ydraws))  # sd ≈ sqrt(σ² + σ_g²-ish)
    @test length(ydraws) > 100      # guards extraction shape before the real checks below
    @test isapprox(mean(ydraws), μ̂; atol = 0.2)
    @test two_sample_ks(ydraws, ref)
end

@testset "prior predictive sanity" begin
    m = model(
        (@likelihood Normal y ~ x), @priors(b ~ Normal(0, 1)),
        (x = [0.1, 0.2], y = [1.0, 2.0])
    )
    pchain = Logging.with_logger(Logging.NullLogger()) do
        sample(StableRNG(3), m, Prior(), 200; progress = false)
    end
    FC = parentmodule(typeof(pchain))
    @test FC.niters(pchain) == 200
    pnames = string.(FC.parameters(pchain))
    @test all(nm -> nm in pnames, ["intercept", "b", "sigma"])
end

@testset "recovery: categorical + continuous + interaction, labeled access" begin
    rng = StableRNG(468)
    n = 4000
    grp = rand(rng, ["lo", "hi"], n)
    x = randn(rng, n)
    d = Int.(grp .== "hi")
    # y = 1.0 + 0.5*hi + 2.0*x - 1.5*hi*x + eps
    y = 1.0 .+ 0.5 .* d .+ 2.0 .* x .- 1.5 .* d .* x .+ 0.3 .* randn(rng, n)
    df = (y = y, grp = grp, x = x)
    lik = @likelihood Normal y ~ grp * x
    pri = @priors begin
        intercept ~ Normal(0, 5)
        b ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    m = model(lik, pri, df)
    chn = sample(StableRNG(469), m, NUTS(), 500; progress = false)
    b_draws = chn[@varname(b), stack = true]
    labels = collect(DimensionalData.lookup(b_draws, :coef))
    # Which level StatsModels picks as dummy-coding reference decides the
    # parameterization. Truth: y = 1 + 0.5·[hi] + 2x − 1.5·[hi]x.
    # Reference "hi" (dummy grp_lo): y = 1.5 − 0.5·[lo] + 0.5x + 1.5·[lo]x.
    # Both branches are exact algebra of the same truth — either passes.
    if :grp_lo in labels
        @test mean(chn[@varname(intercept)]) ≈ 1.5 atol = 0.15
        @test mean(b_draws[coef = At(:grp_lo)]) ≈ -0.5 atol = 0.15
        @test mean(b_draws[coef = At(:x)]) ≈ 0.5 atol = 0.15
        @test mean(b_draws[coef = At(:grp_lo__x)]) ≈ 1.5 atol = 0.15
    else
        @test :grp_hi in labels
        @test mean(chn[@varname(intercept)]) ≈ 1.0 atol = 0.15
        @test mean(b_draws[coef = At(:grp_hi)]) ≈ 0.5 atol = 0.15
        @test mean(b_draws[coef = At(:x)]) ≈ 2.0 atol = 0.15
        @test mean(b_draws[coef = At(:grp_hi__x)]) ≈ -1.5 atol = 0.15
    end
end
