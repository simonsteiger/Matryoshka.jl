using Matryoshka
using Matryoshka: Intercept, FixedEffects, RandomIntercept,
    compprefix, submodel, priorslots, rebuild
using Distributions, DynamicPPL, Turing, Test

@testset "Intercept" begin
    c = Intercept(4)
    @test compprefix(c) === nothing
    m = submodel(c, Normal(0, 10))
    @model hand() = intercept ~ Normal(0, 10)
    @test logjoint(m, (intercept = 1.3,)) ≈ logjoint(hand(), (intercept = 1.3,))
    retval, _ = DynamicPPL.evaluate!!(DynamicPPL.condition(m, intercept = 2.0), DynamicPPL.VarInfo())
    @test retval == fill(2.0, 4)
    slots = priorslots(c)
    @test length(slots) == 1
    @test slots[1][1] == (:intercept,) && slots[1][2] == (:intercept,)
end

@testset "FixedEffects" begin
    X = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    c = FixedEffects(X, [:x1, :x2], nothing)
    m = submodel(c, [Normal(0, 1), Normal(0, 5)])
    slots = priorslots(c)
    @test [s[1] for s in slots] == [(:b, :x1), (:b, :x2)]
    @test all(s -> s[2] == (:b,), slots)
    # contribution check: condition coefficients, evaluate return value
    # spike Q3 verdict: arraydist fallback -> single vector VarName `b`
    b = [0.5, -1.0]
    retval, _ = DynamicPPL.evaluate!!(DynamicPPL.condition(m, b = b), DynamicPPL.VarInfo())
    @test retval == X * b
    @model hand() = b ~ arraydist([Normal(0, 1), Normal(0, 5)])
    @test logjoint(m, (b = b,)) ≈ logjoint(hand(), (b = b,))
end

@testset "RandomIntercept" begin
    c = RandomIntercept(:g, [1, 1, 2, 3], ["a", "b", "c"])
    @test compprefix(c) === :g
    m = submodel(c, Exponential(1))
    @model function hand(idx, n)
        sd ~ Exponential(1)
        z ~ filldist(Normal(), n)
        return (sd .* z)[idx]
    end
    θ = (sd = 0.7, z = [0.1, -0.2, 0.5])
    @test logjoint(m, θ) ≈ logjoint(hand([1, 1, 2, 3], 3), θ)
    @test priorslots(c)[1][1] == (:g, :sd)
    @test priorslots(c)[1][2] == (:sd,)
end
