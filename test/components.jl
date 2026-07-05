using Matryoshka
using Matryoshka: Intercept, FixedEffects, RandomIntercept,
    compprefix, submodel, priorslots, rebuild
using Distributions, DynamicPPL, Turing, Test
using StatsModels: StatsModels
using Tables: Tables
using DimensionalData: DimensionalData, At

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
    # spike Q3 verdict: product_distribution fallback -> single vector VarName `b`
    b = [0.5, -1.0]
    retval, _ = DynamicPPL.evaluate!!(DynamicPPL.condition(m, b = b), DynamicPPL.VarInfo())
    @test retval == X * b
    @model hand() = b ~ product_distribution([Normal(0, 1), Normal(0, 5)])
    @test logjoint(m, (b = b,)) ≈ logjoint(hand(), (b = b,))
    # labeled draws: b is a DimVector on dim :coef with the component's names
    draw = NamedTuple(rand(m))
    @test draw.b isa DimensionalData.AbstractDimVector
    @test DimensionalData.hasdim(draw.b, :coef)
    @test collect(DimensionalData.lookup(draw.b, :coef)) == [:x1, :x2]
    @test draw.b[At(:x1)] isa Float64
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
    # labeled draws: z is a DimVector on a dim named after the grouping
    # variable, labels are sanitized levels
    draw = NamedTuple(rand(m))
    @test draw.z isa DimensionalData.AbstractDimVector
    @test DimensionalData.hasdim(draw.z, :g)
    @test collect(DimensionalData.lookup(draw.z, :g)) == [:a, :b, :c]
    @test draw.z[At(:b)] isa Float64
end

@testset "rebuild" begin
    @testset "Intercept" begin
        c2 = rebuild(Intercept(4), (y = [1.0, 2.0],))
        @test c2 isa Intercept
        @test c2.n == 2
    end

    @testset "RandomIntercept" begin
        c = RandomIntercept(:g, [1, 1, 2, 3], ["a", "b", "c"])
        c2 = rebuild(c, (g = ["c", "a", "a"],))
        @test c2.idx == [3, 1, 1]
        @test c2.group === :g
        @test c2.levels == ["a", "b", "c"]
        @test_throws ArgumentError rebuild(c, (g = ["a", "d"],))
        err = try
            rebuild(c, (g = ["a", "d"],))
        catch e
            e
        end
        @test occursin(":g", err.msg)
        @test occursin("not yet supported", err.msg)
    end

    @testset "FixedEffects" begin
        tbl0 = (y = [1.0, 2.0, 3.0], x1 = [0.5, 1.5, 2.5], x2 = [1.0, 0.0, 1.0])
        f = StatsModels.@formula(y ~ 0 + x1 + x2)
        sch = StatsModels.apply_schema(f, StatsModels.schema(f, tbl0), StatsModels.StatisticalModel)
        term = sch.rhs
        c = FixedEffects(Matrix{Float64}(StatsModels.modelcols(term, tbl0)), [:x1, :x2], term)
        tbl1 = (y = [0.0, 0.0], x1 = [2.0, 4.0], x2 = [1.0, 1.0])
        c2 = rebuild(c, tbl1)
        @test c2.X == [2.0 1.0; 4.0 1.0]
        @test c2.X == Matrix{Float64}(StatsModels.modelcols(term, Tables.columntable(tbl1)))
        @test c2.names == [:x1, :x2]
        @test c2.term === term
    end
end
