using DynamicPPL, Matryoshka, Distributions, Test
using DynamicPPL.TestUtils.AD: run_ad
using ADTypes: AutoForwardDiff, AutoReverseDiff, AutoMooncake
import ForwardDiff, ReverseDiff, Mooncake

df = (y = [1.0, 2.0, 0.5, 1.4], x = [0.1, 0.9, 0.4, 0.7], g = ["a", "b", "a", "b"])
models = [
    model((@likelihood Normal y ~ x), @priors(b ~ Normal(0, 1)), df),
    model((@likelihood Normal y ~ x + (1 | g)), @priors(b ~ Normal(0, 1)), df),
    model(
        (@likelihood Poisson y ~ x), @priors(b ~ Normal(0, 1)),
        (y = [0, 2, 1, 4], x = df.x)
    ),
]
adtypes = [AutoForwardDiff(), AutoReverseDiff(), AutoMooncake(; config = nothing)]

@testset "AD grid" begin
    for m in models, ad in adtypes
        @testset "$(nameof(typeof(ad)))" begin
            run_ad(m, ad)   # throws on failure
        end
    end
end
