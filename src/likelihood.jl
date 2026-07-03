"""
    Likelihood{F,T}

Parsed output of `@likelihood`: `family::F` (a `Family` instance) paired with
`formulas::T`, a tuple of `StatsModels.FormulaTerm`s — the response formula
first, any distributional-regression formulas after. Construct via the
`@likelihood` macro, not directly.

# Example
```julia
using Matryoshka

lik = @likelihood Normal y ~ x
lik.family isa NormalFamily
```
"""
struct Likelihood{F <: Family, T <: Tuple}
    family::F
    formulas::T
end

response_formula(lik::Likelihood) = first(lik.formulas)
extra_formulas(lik::Likelihood) = Base.tail(lik.formulas)

function formula_line(ex)
    Meta.isexpr(ex, :call, 3) && ex.args[1] === :~ ||
        error("@likelihood: each line must be `lhs ~ rhs`, got `$(ex)`")
    # Delegate to StatsModels' own parser so (1 | g) etc. parse identically to @formula.
    # Build the macrocall with a GlobalRef to StatsModels' @formula (rather than the
    # unqualified `StatsModels.@formula(...)` quoted form) so it resolves even when the
    # caller's module has no `StatsModels` binding of its own — `esc(...)` only protects
    # hygiene, it does not import bindings into the caller's scope.
    return Expr(
        :macrocall, GlobalRef(StatsModels, Symbol("@formula")),
        LineNumberNode(@__LINE__, Symbol(@__FILE__)), :($(ex.args[2]) ~ $(ex.args[3])),
    )
end

"""
    @likelihood(family, ex)

Build a `Likelihood` spec from a `family` and one or more tilde-formula lines.

`family` names the response family as a `Distributions.jl` distribution type
(`Normal`, `Bernoulli`, `Poisson`). `ex` is either a single `lhs ~ rhs`
expression or a `begin...end` block of them: the first line is the response
formula (e.g. `y ~ x + (1 | g)`); further lines target other family parameters
(e.g. `sigma ~ z`) — supported by the design but not yet implemented in v0
(`model()` raises an `ArgumentError` if any are present). Term parsing
delegates to `StatsModels.@formula`, so formula syntax (`+`, `(1 | g)`, `0`,
...) matches StatsModels exactly.

# Example
```julia
using Matryoshka

lik = @likelihood Normal begin
    y ~ x + (1 | g)
end
```
"""
macro likelihood(family, ex)
    lines = Meta.isexpr(ex, :block) ? filter(x -> !(x isa LineNumberNode), ex.args) : [ex]
    isempty(lines) && error("@likelihood: needs at least a response formula")
    formulas = map(formula_line, lines)
    return :(Likelihood(to_family($(esc(family))), ($(map(esc, formulas)...),)))
end
