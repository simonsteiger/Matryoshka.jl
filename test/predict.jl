using Matryoshka
using Distributions, DynamicPPL, Turing, StableRNGs, Test
using DimensionalData: DimensionalData
using Logging: Logging

df = (
    y = [1.1, 2.3, 0.9, 1.8, 2.5, 1.2],
    x = [0.5, 1.0, 0.2, 0.9, 1.4, 0.3],
    g = ["a", "a", "b", "b", "c", "c"],
)

@testset "rebuild + predict" begin
    lik = @likelihood Normal y ~ x + (1 | g)
    pri = @priors b ~ Normal(0, 1)
    m = model(lik, pri, df)
    chain = Logging.with_logger(Logging.NullLogger()) do
        sample(StableRNG(1), m, NUTS(), 300; progress = false)
    end

    newdata = (x = [0.4, 1.2], g = ["a", "c"])       # no y column → y missing
    m_new = model(m, newdata)
    @test m_new isa DynamicPPL.Model
    @test m_new.args.y === missing

    preds = predict(StableRNG(2), m_new, chain)
    # Turing >= 0.45 returns FlexiChains, not MCMCChains (test/spike_notes.md).
    # `y` is a single vector-valued VarName (spike Q3 deviation): each posterior
    # predictive draw is a length-2 vector, one entry per newdata row.
    FC = parentmodule(typeof(preds))
    @test @varname(y) in FC.parameters(preds)
    # `y` draws are now Dim{:obs}-labeled DimVectors (Task 6), so plain
    # `preds[vn]` would implicitly auto-stack them into one 3D DimArray
    # (iter, chain, obs) — pass `stack = false` explicitly to keep the
    # original per-draw-vector shape this block asserts on.
    yv = preds[@varname(y), stack = false]
    @test size(yv, 1) == FC.niters(preds)
    @test all(v -> length(v) == 2, yv)

    # predictions are obs-labeled, sized to newdata (2 rows in `newdata` above)
    y_draws = preds[@varname(y), stack = true]
    @test DimensionalData.hasdim(y_draws, :obs)
    @test length(DimensionalData.lookup(y_draws, :obs)) == 2

    # new factor level errors in domain language
    err = try
        model(m, (x = [1.0], g = ["zzz"]))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("new level", err.msg) && occursin(":g", err.msg)

    # missing formula column in newdata errors in domain language (check_columns,
    # shared with `lower`), not an opaque StatsModels/FieldError
    err_col = try
        model(m, (g = ["a", "c"],))    # no x column
        nothing
    catch e
        e
    end
    @test err_col isa ArgumentError
    @test occursin(":x", err_col.msg) && occursin("available columns", err_col.msg)

    # response present → observed rebuild (refit workflow)
    m_refit = model(m, df)
    @test m_refit.args.y == df.y
    θ = (intercept = 0.0, b = [0.5], sigma = 1.0, g = (sd = 1.0, z = [0.1, 0.2, 0.3]))
    @test logjoint(m_refit, θ) isa Real
end
