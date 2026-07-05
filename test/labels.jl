using Matryoshka
using Distributions, DynamicPPL, Turing, Test
using DimensionalData: DimensionalData, Dim, At
using DimensionalDistributions: withdims

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
