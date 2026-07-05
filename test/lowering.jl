using Matryoshka
using Matryoshka: lower, FixedEffects, RandomIntercept, Intercept
using Distributions, Test
using StatsModels: StatsModels

@testset "lower: interactions survive (v0 bug regression)" begin
    df = (y = randn(10), x = randn(10), z = randn(10))
    lik = @likelihood Normal y ~ x * z
    comps, _, _ = lower(lik, df)
    fe = comps[end]
    @test fe isa FixedEffects
    @test fe.names == [:x, :z, :x__z]
    @test size(fe.X) == (10, 3)
    @test fe.X[:, 3] ≈ fe.X[:, 1] .* fe.X[:, 2]
end

@testset "lower: sanitized categorical + interaction + group term" begin
    df = (
        y = randn(12),
        species = repeat(["Adelie", "Chinstrap", "Gentoo"], 4),
        body_mass_g = randn(12),
        g = repeat(["u", "v"], 6),
    )
    lik = @likelihood Normal y ~ species * body_mass_g + (1 | g)
    comps, _, _ = lower(lik, df)
    @test any(c -> c isa Intercept, comps)
    @test any(c -> c isa RandomIntercept, comps)
    fe = comps[findfirst(c -> c isa FixedEffects, collect(comps))]
    @test fe.names == [
        :species_Chinstrap, :species_Gentoo, :body_mass_g,
        :species_Chinstrap__body_mass_g, :species_Gentoo__body_mass_g,
    ]
end

@testset "lower: sanitized-name collision errors" begin
    # column literally named like the sanitized dummy of another column
    df = (
        y = randn(9),
        species = repeat(["Adelie", "Chinstrap", "Gentoo"], 3),
        species_Gentoo = randn(9),
    )
    lik = @likelihood Normal y ~ species + species_Gentoo
    @test_throws ArgumentError lower(lik, df)
    err = try
        lower(lik, df)
    catch e
        e
    end
    @test occursin("species_Gentoo", err.msg)
    @test occursin("Rename", err.msg)
end

@testset "lower: group-level collision errors" begin
    # two levels sanitizing to the same label
    df = (y = randn(4), g = ["a b", "ab", "a b", "ab"])
    lik = @likelihood Normal y ~ (1 | g)
    @test_throws ArgumentError lower(lik, df)
end
