using Matryoshka
using Documenter, DocumenterVitepress

DocMeta.setdocmeta!(Matryoshka, :DocTestSetup, :(using Matryoshka); recursive = true)

makedocs(;
    sitename = "Matryoshka.jl",
    authors = "Simon Steiger",
    modules = [Matryoshka],
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/simonsteiger/Matryoshka.jl",
        devbranch = "main",
        devurl = "dev",
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
