module MatryoshkaTestUtils

using Test, Statistics, HypothesisTests

"""
    resolve_param(chain, nm::AbstractString)

Look up posterior draws for parameter `nm` in a FlexiChains chain
(Turing >= 0.45 returns FlexiChains, not MCMCChains — see test/spike_notes.md
and the idioms already used in test/model.jl / test/predict.jl).

Scalar VarNames (e.g. `"intercept"`, `"sigma"`, prefixed submodel names like
`"g.sd"`) are matched against `string(vn)` for `vn in parameters(chain)`.
Vector components written `"b[1]"` resolve the vector VarName `b` and pull
out the requested index from each draw — Matryoshka's slope prior is a
single `arraydist` VarName `b`, not scalar `b[1]`/`b[2]` VarNames (spike
verdict).
"""
function resolve_param(chain, nm::AbstractString)
    FC = parentmodule(typeof(chain))
    vns = FC.parameters(chain)
    m = match(r"^(.+)\[(\d+)\]$", nm)
    if m === nothing
        vn = only(filter(v -> string(v) == nm, vns))
        return chain[vn]
    else
        base, idx = m.captures[1], parse(Int, m.captures[2])
        vn = only(filter(v -> string(v) == base, vns))
        return getindex.(chain[vn], idx)
    end
end

function check_numerical(chain, names_vals::Vector{<:Pair}; atol = 0.2)
    for (nm, val) in names_vals
        draws = resolve_param(chain, nm)
        est = mean(draws)
        @info "recovery" name = nm exact = val estimate = est
        @test est ≈ val atol = atol
    end
    return
end

function two_sample_ks(a, b; α = 1.0e-3)
    return HypothesisTests.pvalue(HypothesisTests.ApproximateTwoSampleKSTest(vec(a), vec(b))) > α
end

end
