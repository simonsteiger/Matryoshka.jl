# DynamicPPL / Turing submodel mechanics — spike results

Environment: Turing v0.45.0, DynamicPPL v0.41.8, Distributions v0.25.129, Julia 1.10.10.
Chains returned by `sample`/`predict` are **FlexiChains** (not MCMCChains) — `keys(chain)`
returns `Parameter(varname)` / `Extra(:name)` entries; a `Parameter` can hold a
vector-valued VarName as a single entry (see Q3).

All snippets below were run verbatim in the project REPL. Note: `using Turing`
loads DynamicPPL internally but does not bind the module name `DynamicPPL` in the
current scope, so qualified calls like `DynamicPPL.prefix(...)` fail with
`UndefVarError` until you also run `using DynamicPPL`.

## Q1 — manual prefixing API

**Verdict: WORKS** (primary incantation, no fallback needed).

```julia
using Turing
using DynamicPPL

@model inner() = (a ~ Normal(); return a)
@model function outer(m)
    x ~ to_submodel(m, false)
    b ~ Normal(x)
end
m = outer(DynamicPPL.prefix(inner(), :g))
keys(VarInfo(m))
```

Observed output:

```julia
2-element Vector{VarName}:
 g.a
 b
```

`DynamicPPL.prefix(model, ::Symbol)` exists and works directly in v0.41.8. No need
for `Val(:g)` or the `to_submodel(inner(), true)` LHS-naming fallback.

## Q2 — tuple-recursive submodel composition

**Verdict: WORKS** (primary incantation, no fallback needed).

```julia
using Turing
using DynamicPPL

@model function sum_contribs(ms::Tuple)
    c ~ to_submodel(first(ms), false)
    if length(ms) == 1
        return c
    end
    rest ~ to_submodel(sum_contribs(Base.tail(ms)), false)
    return c .+ rest
end
@model contrib(v) = (u ~ Normal(); return fill(u, 3) .+ v)
@model function outer2()
    η ~ to_submodel(sum_contribs((DynamicPPL.prefix(contrib(0.0), :p1), DynamicPPL.prefix(contrib(1.0), :p2))), false)
    y ~ MvNormal(η, I)
end
rand(outer2())
ch2 = sample(outer2(), NUTS(), 50; progress=false)
keys(ch2)
```

`rand(outer2())` returned a `VarNamedTuple` with `p1.u`, `p2.u`, `y` — confirming
recursive `to_submodel` of the same model function works without unrolling.

Observed `keys(ch2)` (FlexiChains KeySet, 17 entries: 3 params + 14 sampler extras):

```
Parameter(p1.u)
Parameter(p2.u)
Parameter(y)
Extra(:n_steps) ... (14 sampler-stat extras omitted)
```

Recursive `to_submodel` of the same model function (`sum_contribs` calling itself
via `Base.tail`) works as written — no pairwise-unroll fallback needed.

## Q3 — coefficient naming (Dict-indexed vs arraydist)

**Verdict: PRIMARY FAILS, FALLBACK WORKS — naming invariant deviates from brief's
expectation.**

Primary (Dict-style symbol indexing) fails:

```julia
@model function fe1(names, priors)
    b = Dict{Symbol,Real}()
    for (n, p) in zip(names, priors)
        b[n] ~ p
    end
    return nothing
end
sample(fe1([:x1, :x2], [Normal(), Normal()]), NUTS(), 50; progress=false)
```

Failure message:

```
ERROR: DynamicPPL currently does not support random variables within any container
that is not an `AbstractArray` (got type Dict{Symbol, Real}). For example, if `x`
is a `Dict` and you have `x["a"] ~ dist`, this will fail.
```

Fallback (`arraydist`) works, but **does not** produce per-element VarNames
(`b[1]`, `b[2]`) as the brief anticipated (that expectation was based on
MCMCChains column-naming). Under DynamicPPL 0.41 / FlexiChains, `arraydist`
assignment creates a single VarName `b` whose value is the whole vector:

```julia
@model function fe1_fallback(names, priors)
    b ~ arraydist(priors)
    return nothing
end
ch3b = sample(fe1_fallback([:x1, :x2], [Normal(), Normal()]), NUTS(), 50; progress=false)
keys(ch3b)
keys(VarInfo(fe1_fallback([:x1, :x2], [Normal(), Normal()])))
```

Observed:

```julia
# keys(VarInfo(...))
1-element Vector{VarName}:
 b

# ch3b chain summary
Parameters (1) ── VarName
 Vector{Float64}  b
```

**Deviation to record for Tasks 5–7:** there is no `b[1]`/`b[2]` (or `b[:x1]`/`b[:x2]`)
VarName-level naming available from `arraydist` in this DynamicPPL/FlexiChains
version — the whole vector is stored under one VarName `b`. Any coefficient-name
mapping (`:x1` → index 1, `:x2` → index 2) must be tracked externally (e.g. a
`coefnames` vector kept alongside the model) and applied by *positional indexing
into the vector-valued `b` column after extraction*, not by looking up per-name
chain columns.

## Q4 — predict with y-as-argument

**Verdict: WORKS**, no adjustment needed — scalar `missing` (not
`Vector{Union{Missing,Float64}}`) is sufficient.

```julia
@model obs(mu, y) = y ~ product_distribution(Normal.(mu, 1.0))
@model function full(y)
    m ~ Normal()
    v ~ to_submodel(obs(fill(m, 4), y), false)
end
ch4 = sample(full([1.0, 1.1, 0.9, 1.0]), NUTS(), 100; progress=false)
pr = predict(full(missing), ch4)
pr[:y][1, 1]
```

Observed: `predict` ran without error and returned a FlexiChain with parameters
`m` and `y` (`Vector{Float64}`, length 4 per draw — matching `product_distribution`'s
dimensionality even though the model argument passed was a bare scalar `missing`,
not a pre-sized `Vector{Union{Missing,Float64}}(missing, 4)`):

```julia
4-element Vector{Float64}:
  1.3025017299751203
  1.5162735171448962
 -0.3472769837194991
  0.22206408222875862
```

## Summary for Tasks 5–7

| Q | Verdict | Adopt |
|---|---------|-------|
| Q1 | Works as written | `DynamicPPL.prefix(model, :sym)` |
| Q2 | Works as written | recursive tuple `to_submodel` over `Base.tail` |
| Q3 | Primary fails; fallback works with deviation | `b ~ arraydist(priors)`, single VarName `b` holds full vector — track coefficient names externally, index positionally |
| Q4 | Works as written | scalar `missing` as model arg is enough; no need to pre-size a `Vector{Union{Missing,T}}` |
