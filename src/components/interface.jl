"""
    AbstractComponent

Abstract supertype for predictor-term components (`Intercept`, `FixedEffects`,
`RandomIntercept`). A component holds the data-contact artifacts for one
formula term (design matrix, group indices, factor levels) and implements four
functions: `compprefix`, `submodel`, `priorslots`, `rebuild`. New term types
(e.g. Gaussian-process or smooth terms) are added by subtyping
`AbstractComponent`, implementing these four, and adding a `lower` dispatch
(see `src/lowering.jl`) that turns a matching formula term into the component.
New term types should label vector-valued parameters by wrapping their prior in
`withdims` (see `FixedEffects`/`RandomIntercept` for the pattern): fixed-effect
coefficients use the fixed `:coef` dim, group-level parameters a dim named after
the grouping variable.

# Example
```julia
using Matryoshka
using Matryoshka: Intercept

Intercept(4) isa AbstractComponent
```
"""
abstract type AbstractComponent end

"""
    compprefix(c::AbstractComponent) -> Union{Symbol,Nothing}

Namespace prefix for `c`'s submodel, or `nothing` for an unprefixed submodel.
`RandomIntercept` returns its grouping variable (e.g. `:g`, so its parameters
appear as `g.sd`, `g.z`); `Intercept` and `FixedEffects` return `nothing`.

# Example
```julia
using Matryoshka
using Matryoshka: RandomIntercept

compprefix(RandomIntercept(:g, [1, 2, 1], ["a", "b"])) === :g
```
"""
function compprefix end

"""
    submodel(c::AbstractComponent, prior) -> DynamicPPL.Model

Build `c`'s own `@model`, parameterised by `prior` (the resolved prior, or
vector of priors, for `c`'s slot(s) — as targeted by `priorslots`). The
submodel samples its own parameter(s) and returns its n-vector contribution to
the linear predictor. Only draws sampled via `~` may carry `withdims` labels;
the n-vector contribution returned to `core_model` must be a plain, unlabeled
array (`collect` it first if it was derived from a labeled `DimArray`).

# Example
```julia
using Matryoshka, Distributions, DynamicPPL
using Matryoshka: Intercept

c = Intercept(4)
m = submodel(c, Normal(0, 10))
m isa DynamicPPL.Model
```
"""
function submodel end

"""
    priorslots(c::AbstractComponent) -> Vector{<:Tuple}

Prior-targeting slots for `c`, one per parameter `c` introduces. Each slot is
an `(exact, class, default)` triple: `exact` and `class` are
`Tuple{Vararg{Symbol}}` prior-target paths (dotted-name and bare-name forms,
e.g. `(:b, :x)` and `(:b,)`), and `default` is the `Distributions.jl` prior
used when no `@priors` line matches either. Consumed by `resolve_priors` at
`model()` time, and by `default_priors(lik, tbl)` for introspection.

# Example
```julia
using Matryoshka, Distributions
using Matryoshka: RandomIntercept

priorslots(RandomIntercept(:g, [1, 2, 1], ["a", "b"])) == [((:g, :sd), (:sd,), Exponential(1))]
```
"""
function priorslots end

"""
    rebuild(c::AbstractComponent, tbl) -> AbstractComponent

Recompute `c` against new data `tbl`, keeping training-time artifacts (factor
levels, contrasts) fixed. Used by `model(m, newdata)` to rebuild each
component for prediction or refitting. Grouping variables reject unseen factor
levels with an `ArgumentError` (`allow_new_levels` is not yet supported).

# Example
```julia
using Matryoshka
using Matryoshka: Intercept

rebuild(Intercept(4), (y = [1.0, 2.0],)).n == 2
```
"""
function rebuild end
