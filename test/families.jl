using Matryoshka
using Matryoshka: NormalFamily, BernoulliFamily, PoissonFamily,
    parameters, links, invlink, default_priors, obsmodel, to_family
using Distributions, DynamicPPL, Turing
using Test

@testset "family interface" begin
    f = NormalFamily()
    @test parameters(f) == (:mu, :sigma)
    @test links(f) == (mu = :identity, sigma = :log)
    @test invlink(Val(:log)) === exp
    @test default_priors(f) == (sigma = Exponential(1),)
    @test to_family(Normal) === NormalFamily()
    @test to_family(Poisson) === PoissonFamily()
    @test parameters(BernoulliFamily()) == (:mu,)
    @test links(PoissonFamily()) == (mu = :log,)
end

@testset "obsmodel logjoint equality vs handwritten" begin
    y = [0.5, 1.2, -0.3]
    mu = [0.4, 1.0, 0.0]
    m = obsmodel(NormalFamily())(mu, (sigma = Exponential(1),), y)
    @model function hand(mu, y)
        sigma ~ Exponential(1)
        y ~ product_distribution(Normal.(mu, sigma))
    end
    for sigma in (0.5, 1.0, 2.0)
        @test logjoint(m, (; sigma)) ≈ logjoint(hand(mu, y), (; sigma))
    end

    yb = [0, 1, 1]
    mb = obsmodel(BernoulliFamily())([0.2, 0.7, 0.9], NamedTuple(), yb)
    @model handb(p, y) = y ~ product_distribution(Bernoulli.(p))
    @test logjoint(mb, NamedTuple()) ≈ logjoint(handb([0.2, 0.7, 0.9], yb), NamedTuple())

    yp = [0, 2, 5]
    mp = obsmodel(PoissonFamily())([0.5, 1.0, 4.0], NamedTuple(), yp)
    @model handp(l, y) = y ~ product_distribution(Poisson.(l))
    @test logjoint(mp, NamedTuple()) ≈ logjoint(handp([0.5, 1.0, 4.0], yp), NamedTuple())
end
