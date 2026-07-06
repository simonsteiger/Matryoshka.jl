using JET, Matryoshka, Distributions, DynamicPPL, Test

# All-sites fixture: `x * z` exercises Dim{:coef} with an interaction label
# (b.x__z), `(1 | g)` exercises the group dim, and the Normal obsmodel
# exercises Dim{:obs}.
df = (y = [1.0, 2.0, 0.5, 1.4], x = [0.1, 0.9, 0.4, 0.7], z = [0.3, 0.6, 0.2, 0.9], g = ["a", "b", "a", "b"])
lik = @likelihood Normal y ~ x * z + (1 | g)
pri = @priors b ~ Normal(0, 1)

# Known caveat: JET may fail on the submodel wiring until DynamicPPL v0.42.1
# (upstream submodel bugs, fixed there). If a failure here also reproduces on
# main (pre-labels), it is upstream — record, do not debug in this package.
@testset "JET" begin
    m = model(lik, pri, df)
    # @test_opt fails structurally (StatsModels construction-time dispatch; DynamicPPL
    # recursive to_submodel) — see .claude/notes/reports/2026-07-03-jet-test-opt-limitations.md;
    # revisit after DynamicPPL upgrades.
    JET.@test_call target_modules = (Matryoshka,) Matryoshka.lower(lik, df)
    JET.@test_call target_modules = (Matryoshka,) DynamicPPL.logjoint(m, DynamicPPL.VarInfo(m))
end
