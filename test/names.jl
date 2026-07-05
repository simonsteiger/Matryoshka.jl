using Matryoshka: sanitize, sanitize_level, check_unique_labels
using Test

@testset "sanitize" begin
    # continuous: pass through
    @test sanitize("body_mass_g") == :body_mass_g
    # categorical dummy: ": " → "_"
    @test sanitize("species: Gentoo") == :species_Gentoo
    # interaction: " & " → "__"
    @test sanitize("x & z") == :x__z
    @test sanitize("species: Gentoo & body_mass_g") == :species_Gentoo__body_mass_g
    # three-way interaction
    @test sanitize("a & b & c") == :a__b__c
    # level stripped to identifier-safe
    @test sanitize("g: Very High") == :g_VeryHigh
    @test sanitize_level("Very High") == "VeryHigh"
    @test sanitize_level("a-b.c") == "abc"
end

@testset "check_unique_labels" begin
    @test check_unique_labels([:a, :b], ["a", "b"], "coefficient") === nothing
    # collision: sanitized names merge, raw origins listed
    err = try
        check_unique_labels(
            [:species_Gentoo, :species_Gentoo],
            ["species: Gentoo", "species_Gentoo"],
            "coefficient",
        )
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("species_Gentoo", err.msg)
    @test occursin("species: Gentoo", err.msg)   # raw origin shown
    @test occursin("coefficient", err.msg)
    @test occursin("Rename", err.msg)
end
