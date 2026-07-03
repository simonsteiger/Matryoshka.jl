"""
    Family

Abstract supertype for response families (`NormalFamily`, `BernoulliFamily`,
`PoissonFamily`). A family declares its distribution parameters, per-parameter
links, default priors, and observation submodel via `parameters`, `links`,
`default_priors`, and `obsmodel`. New families are added by subtyping `Family`
and implementing these four functions, plus a `to_family` dispatch (from the
`Distributions.jl` type name used as the first argument to `@likelihood`).
"""
abstract type Family end

"""
    parameters(f::Family) -> Tuple{Vararg{Symbol}}

Names of `f`'s distribution parameters, response (predictor-supplied) parameter
first.

# Example
```julia
using Matryoshka

parameters(NormalFamily()) == (:mu, :sigma)
```
"""
function parameters end

"""
    links(f::Family) -> NamedTuple

Default link function for each of `f`'s parameters, keyed by parameter name and
given as a `Symbol` (`:identity`, `:log`, or `:logit`).

# Example
```julia
using Matryoshka

links(PoissonFamily()) == (mu = :log,)
```
"""
function links end

"""
    default_priors(f::Family) -> NamedTuple

Default prior distribution for each of `f`'s parameters that is not supplied by
a formula — e.g. `sigma` for `NormalFamily`, sampled with its own prior unless a
formula such as `sigma ~ z` targets it (distributional regression; not yet
supported in v0).

See also the `default_priors(lik::Likelihood, tbl)` method, which returns the
full resolved-prior table for a model and dataset rather than one family's
defaults alone.

# Example
```julia
using Matryoshka

default_priors(NormalFamily()) == (sigma = Exponential(1),)
```
"""
function default_priors end

"""
    obsmodel(f::Family) -> (eta, priors, y) -> DynamicPPL.Model

Observation-submodel constructor for `f`. Returns a callable built by
`model()`/`model(m, newdata)` as `obsmodel(f)(eta, fam_priors, y)`:

- `eta`: the **link-scale** linear predictor for `f`'s response parameter — the
  sum of all component contributions, *before* any inverse link is applied.
  Not the mean.
- `fam_priors`: a `NamedTuple` of resolved priors for `f`'s other parameters
  (e.g. `(sigma = Exponential(1),)` for `NormalFamily`).
- `y`: the observed response, or `missing` when predicting.

Each family applies its own inverse link to `eta` inside the returned model
(not by first computing a mean-scale parameter), which is numerically safer
under autodiff:

| Family            | uses `eta` as                       |
|:------------------|:-------------------------------------|
| `NormalFamily`    | `mu` directly (`identity` link)      |
| `BernoulliFamily` | `BernoulliLogit.(eta)` (logit link)  |
| `PoissonFamily`   | `Poisson.(exp.(eta))` (log link)     |

# Example
```julia
using Matryoshka, Distributions, DynamicPPL

eta = [0.1, -0.2, 0.3]
y = [0, 1, 0]
m = obsmodel(BernoulliFamily())(eta, NamedTuple(), y)
m isa DynamicPPL.Model
```
"""
function obsmodel end

invlink(::Val{:identity}) = identity
invlink(::Val{:log}) = exp
invlink(::Val{:logit}) = logistic
logistic(x) = inv(one(x) + exp(-x))

to_family(f::Family) = f
predictor_param(f::Family) = first(parameters(f))
predictor_invlink(f::Family) = invlink(Val(links(f)[predictor_param(f)]))
