using Matryoshka
using Documenter, DocumenterVitepress
using PalmerPenguins, DataFrames
using Turing, FlexiChains
using CairoMakie

DocMeta.setdocmeta!(Matryoshka, :DocTestSetup, :(using Matryoshka); recursive = true)

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

makedocs(;
    sitename = "Matryoshka.jl",
    authors = "Simon Steiger",
    modules = [Matryoshka],
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/simonsteiger/Matryoshka.jl",
        devbranch = "main",
        devurl = "dev",
        # build_vitepress = false,
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
    warnonly = true,
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/simonsteiger/Matryoshka.jl",
    target = joinpath(@__DIR__, "build"),
    devbranch = "main",
    push_preview = true,
)
