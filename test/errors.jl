using Matryoshka, Distributions, Test

df = (y = [1.0, 2.0], x = [0.1, 0.2], g = ["a", "b"])

@testset "model()-time errors, domain language" begin
    lik = @likelihood Normal y ~ x

    # unknown prior target lists valid ones
    bad = @priors begin
        b.z ~ Normal(0, 1)
    end
    err = try
        model(lik, bad, df); nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("unknown prior target", err.msg)
    @test occursin("b.x", err.msg)          # valid targets listed

    # missing column
    lik2 = @likelihood Normal y ~ w
    err2 = try
        model(lik2, @priors(b ~ Normal(0, 1)), df); nothing
    catch e
        e
    end
    @test err2 isa ArgumentError
    @test occursin(":w", err2.msg) && occursin("available columns", err2.msg)

    # distributional formula rejected clearly (v0)
    lik3 = @likelihood Normal begin
        y ~ x
        sigma ~ x
    end
    err3 = try
        model(lik3, @priors(b ~ Normal(0, 1)), df); nothing
    catch e
        e
    end
    @test err3 isa ArgumentError
    @test occursin("distributional regression", err3.msg)

    # random slope rejected clearly (v0)
    lik4 = @likelihood Normal y ~ x + (1 + x | g)
    err4 = try
        model(lik4, @priors(b ~ Normal(0, 1)), df); nothing
    catch e
        e
    end
    @test err4 isa ArgumentError
    @test occursin("random intercepts", err4.msg)

    # empty components: predictor-less formula must fail at model()-time, not at
    # sampling time (sum_contribs(()) would otherwise BoundsError during sampling)
    lik5 = @likelihood Normal y ~ 0
    err5 = try
        model(lik5, @priors(intercept ~ Normal(0, 1)), df); nothing
    catch e
        e
    end
    @test err5 isa ArgumentError
    @test occursin("no predictor", err5.msg)
end
