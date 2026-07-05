# Single source of truth for user-facing parameter labels. StatsModels
# coefnames ("species: Gentoo", "x & z") are not valid identifiers, so they
# cannot serve as prior targets or dim labels; sanitize once, at lower time.
# Rules (see spec): ": " → "_", " & " → "__", levels stripped to
# identifier-safe characters. Continuous names pass through unchanged.

const _INTERACTION_SEP = " & "
const _LEVEL_SEP = ": "

sanitize_level(s::AbstractString) = replace(s, r"[^A-Za-z0-9_]" => "")

function sanitize(coefname::AbstractString)
    parts = map(split(coefname, _INTERACTION_SEP)) do part
        r = findfirst(_LEVEL_SEP, part)
        if r === nothing
            String(part)
        else
            var = part[1:(first(r) - 1)]
            lvl = part[(last(r) + 1):end]
            string(var, "_", sanitize_level(lvl))
        end
    end
    return Symbol(join(parts, "__"))
end

function check_unique_labels(
        labels::AbstractVector{Symbol},
        raw::AbstractVector{<:AbstractString},
        what::AbstractString,
    )
    seen = Dict{Symbol, Vector{String}}()
    for (l, r) in zip(labels, raw)
        push!(get!(seen, l, String[]), String(r))
    end
    dups = sort!([p for p in pairs(seen) if length(p.second) > 1]; by = p -> string(p.first))
    isempty(dups) && return nothing
    desc = join(("`$(p.first)` from [" * join(p.second, ", ") * "]" for p in dups), "; ")
    throw(
        ArgumentError(
            "sanitized $(what) names collide: $desc. " *
                "Rename the offending columns or factor levels so the sanitized names are unique."
        )
    )
end
