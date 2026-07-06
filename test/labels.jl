using Matryoshka
using Distributions, DynamicPPL, Turing, Test
using DimensionalData: DimensionalData, Dim, At
using DimensionalDistributions: withdims
using InferenceObjects: convert_to_inference_data

@testset "spike gate: withdims → NUTS → FlexiChains" begin
    @model function spike()
        x ~ withdims(product_distribution([Normal(0, 1), Normal(0, 2)]), Dim{:coef}([:a, :b]))
        return x
    end
    chn = sample(spike(), NUTS(), 50; progress = false)
    draws = chn[@varname(x), stack = true]
    # (iter, chain, coef) with labeled coef axis
    @test DimensionalData.hasdim(draws, :coef)
    @test collect(DimensionalData.lookup(draws, :coef)) == [:a, :b]
    # label-based selection
    a_draws = draws[coef = At(:a)]
    @test length(a_draws) == 50
end

@testset "end-to-end: penguins-shaped model labels" begin
    n = 60
    df = (
        y = randn(n),
        species = repeat(["Adelie", "Chinstrap", "Gentoo"], n ÷ 3),
        body_mass_g = randn(n),
        pen = repeat(["p1", "p2"], n ÷ 2),
    )
    lik = @likelihood Normal y ~ species + body_mass_g + (1 | pen)
    pri = @priors begin
        intercept ~ Normal(0, 5)
        b ~ Normal(0, 1)
        b.species_Gentoo ~ Normal(0, 5)   # sanitized exact target is writable
        sd ~ Exponential(1)
        sigma ~ Exponential(1)
    end
    m = model(lik, pri, df)
    chn = sample(m, NUTS(), 40; progress = false)

    # invariant: prior target = "b." * dim label, machine-checked
    b_draws = chn[@varname(b), stack = true]
    @test DimensionalData.hasdim(b_draws, :coef)
    coef_labels = collect(DimensionalData.lookup(b_draws, :coef))
    dp = default_priors(lik, df)
    b_targets = [Symbol(r.target[3:end]) for r in dp if startswith(r.target, "b.")]
    @test coef_labels == b_targets

    # random-intercept dim named by grouping variable, labeled by levels
    z_draws = chn[@varname(pen.z), stack = true]
    @test DimensionalData.hasdim(z_draws, :pen)
    @test collect(DimensionalData.lookup(z_draws, :pen)) == [:p1, :p2]

    # label-based selection (canonical robust form)
    @test length(b_draws[coef = At(:body_mass_g)]) == 40

    # InferenceData coord survival (spec integration test). If this fails,
    # do NOT force it green: reduce the assertions to document the actual
    # behavior, leave a comment, and report it as an upstream-issue candidate.
    idata = convert_to_inference_data(chn)
    post = idata.posterior
    @test haskey(post, :b)
    b_names = DimensionalData.name.(DimensionalData.dims(post[:b]))
    @test :coef in b_names
    @test collect(DimensionalData.lookup(post[:b], :coef)) == coef_labels
end

@testset "dim-name conventions are stable" begin
    # :coef and :obs are fixed; group dims are named by the grouping variable.
    # This test freezes the convention (spec section: Naming) — future family
    # parameters disambiguate via VarName prefix (sigma.b), never new dim names.
    #
    # Note: rand(m) on the composed model returns a DynamicPPL `VarNamedTuple`,
    # not a plain `NamedTuple` — property access (`draw.b`) throws a
    # `FieldError`; the working accessor is `draw[@varname(...)]`, verified
    # via kaimon REPL probe against DynamicPPL 0.41.8.
    df = (y = randn(6), x = randn(6), g = repeat(["a", "b"], 3))
    lik = @likelihood Normal y ~ x + (1 | g)
    pri = @priors begin
        intercept ~ Normal(0, 5)
        b ~ Normal(0, 1)
        sd ~ Exponential(1)
        sigma ~ Exponential(1)
    end
    m = model(lik, pri, df)
    draw = rand(m)
    @test DimensionalData.dims(draw[@varname(b)]) isa Tuple{<:DimensionalData.Dim{:coef}}
    @test DimensionalData.dims(draw[@varname(g.z)]) isa Tuple{<:DimensionalData.Dim{:g}}
end
