using Matryoshka
using Matryoshka: Priors, PriorSpec, lookup
using Distributions, Test

@testset "@priors parsing" begin
    p = @priors begin
        intercept ~ TDist(3)
        b ~ Normal(0, 1)
        b.x ~ Normal(0, 5)
        sd ~ Exponential(1)
        g.sd ~ Exponential(0.5)
        sigma ~ Exponential(2)
    end
    @test p isa Priors
    @test length(p.specs) == 6
    @test p.specs[1].path == (:intercept,)
    @test p.specs[3].path == (:b, :x)
    @test p.specs[5].path == (:g, :sd)
    @test p.specs[3].dist == Normal(0, 5)

    q = @priors b ~ Normal(0, 2)   # single-expression form
    @test length(q.specs) == 1 && q.specs[1].path == (:b,)
end

@testset "lookup specificity" begin
    p = @priors begin
        b ~ Normal(0, 1)
        b.x ~ Normal(0, 5)
        sd ~ Exponential(1)
    end
    @test lookup(p, (:b, :x), (:b,)) == Normal(0, 5)     # exact wins
    @test lookup(p, (:b, :z), (:b,)) == Normal(0, 1)     # class fallback
    @test lookup(p, (:g, :sd), (:sd,)) == Exponential(1)
    @test lookup(p, (:h, :tau), (:tau,)) === nothing
end

@testset "@priors rejects non-tilde lines" begin
    @test_throws LoadError @eval @priors begin
        b = Normal(0, 1)
    end
end
