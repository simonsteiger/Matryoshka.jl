using DynamicPPL, Matryoshka, Distributions, Test
using DynamicPPL.TestUtils.AD: run_ad
using ADTypes: AutoForwardDiff
import ForwardDiff
# NOTE: only AutoForwardDiff is supported for now. Reverse-mode backends
# (AutoReverseDiff, AutoMooncake) currently fail: the labeled prior is a
# DimArray, and its parent under reverse-mode is a tracked array whose
# BroadcastStyle (e.g. ReverseDiff.TrackedStyle) is not an AbstractArrayStyle,
# so DimensionalData's `DimensionalStyle(::TrackedStyle)` has no method and the
# MvNormal logpdf broadcast throws. ForwardDiff's Dual is a plain Number, so it
# broadcasts fine. Integrating other AD backends is more involved (fix belongs
# in the DimensionalDistributions fork's AsDimArrayDistribution.logpdf, which
# should strip dims before the numeric logpdf); deferred, not rushed.
# using ADTypes: AutoReverseDiff, AutoMooncake
# import ReverseDiff, Mooncake

df = (
    y = [1.0, 2.0, 0.5, 1.4], x = [0.1, 0.9, 0.4, 0.7], z = [0.3, 0.6, 0.2, 0.9],
    g = ["a", "b", "a", "b"],
)
models = [
    model((@likelihood Normal y ~ x), @priors(b ~ Normal(0, 1)), df),
    # all-sites fixture: `x * z` exercises Dim{:coef} with an interaction
    # label (b.x__z), `(1 | g)` exercises the group dim, and the Normal
    # obsmodel exercises Dim{:obs}.
    model((@likelihood Normal y ~ x * z + (1 | g)), @priors(b ~ Normal(0, 1)), df),
    model(
        (@likelihood Poisson y ~ x), @priors(b ~ Normal(0, 1)),
        (y = [0, 2, 1, 4], x = df.x)
    ),
]
# Reverse-mode backends deferred (see note above); ForwardDiff only for now.
adtypes = [AutoForwardDiff()]
# adtypes = [AutoForwardDiff(), AutoReverseDiff(), AutoMooncake(; config = nothing)]

@testset "AD grid" begin
    for m in models, ad in adtypes
        @testset "$(nameof(typeof(ad)))" begin
            run_ad(m, ad)   # throws on failure
        end
    end
end
