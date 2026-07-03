struct PriorSpec
    path::Tuple{Vararg{Symbol}}
    dist::Distribution
end

"""
    Priors

Parsed output of `@priors`: an ordered list of `PriorSpec(path, dist)` targets.
Construct via the `@priors` macro, not directly. Passed to `model` (or
`default_priors(lik, tbl)`) for resolution against a fitted formula's actual
parameters — exact targets beat class targets, and unmatched targets error
with the list of valid ones.

# Example
```julia
using Matryoshka, Distributions

pri = @priors begin
    intercept ~ TDist(3)
    b ~ Normal(0, 1)
end
pri isa Priors
```
"""
struct Priors
    specs::Vector{PriorSpec}
end

target_path(s::Symbol) = (s,)
function target_path(ex::Expr)
    Meta.isexpr(ex, :., 2) && ex.args[2] isa QuoteNode ||
        error("@priors: left-hand side must be a name or dotted path, got `$(ex)`")
    return (target_path(ex.args[1])..., ex.args[2].value)
end

function prior_line(ex)
    Meta.isexpr(ex, :call, 3) && ex.args[1] === :~ ||
        error("@priors: each line must be `target ~ Distribution`, got `$(ex)`")
    path = target_path(ex.args[2])
    return :(PriorSpec($(QuoteNode(path)), $(esc(ex.args[3]))))
end

"""
    @priors(ex)

Build a `Priors` spec from one or more `target ~ Distribution` lines.

`target` is a bare name (`intercept`, `b`, `sd`, `sigma`, ...) selecting a
*class* of parameters, or a dotted path (`b.x`, `g.sd`) selecting one parameter
*exactly*. `ex` is a single expression or a `begin...end` block. Distributions
are `Distributions.jl` objects — there is no string DSL. Targets are not
checked against a model here; that happens later, in `model()` or
`default_priors(lik, tbl)`, where exact targets override class targets and
unknown targets raise an `ArgumentError` listing the valid ones.

# Example
```julia
using Matryoshka, Distributions

pri = @priors begin
    intercept ~ TDist(3)
    b ~ Normal(0, 1)
    b.x ~ Normal(0, 5)   # overrides the `b` class for coefficient `x`
    g.sd ~ Exponential(0.5)
end
```
"""
macro priors(ex)
    lines = Meta.isexpr(ex, :block) ? filter(x -> !(x isa LineNumberNode), ex.args) : [ex]
    specs = map(prior_line, lines)
    return :(Priors(PriorSpec[$(specs...)]))
end

function lookup(pri::Priors, exact::Tuple, class::Tuple)
    # two-pass: exact beats class; within a pass, the later spec wins
    for s in reverse(pri.specs)
        s.path == exact && return s.dist
    end
    for s in reverse(pri.specs)
        s.path == class && return s.dist
    end
    return nothing
end
