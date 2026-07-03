using JET, Matryoshka, Distributions, DynamicPPL, Test

df = (y = [1.0, 2.0, 0.5], x = [0.1, 0.9, 0.4], g = ["a", "b", "a"])
lik = @likelihood Normal y ~ x + (1 | g)
pri = @priors b ~ Normal(0, 1)

@testset "JET" begin
    m = model(lik, pri, df)
    # @test_opt fails structurally (StatsModels construction-time dispatch; DynamicPPL
    # recursive to_submodel) — see .claude/notes/reports/2026-07-03-jet-test-opt-limitations.md;
    # revisit after DynamicPPL upgrades.
    JET.@test_call target_modules = (Matryoshka,) Matryoshka.lower(lik, df)
    JET.@test_call target_modules = (Matryoshka,) DynamicPPL.logjoint(m, DynamicPPL.VarInfo(m))
end
