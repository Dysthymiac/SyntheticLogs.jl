using SyntheticLogs
using Documenter

DocMeta.setdocmeta!(SyntheticLogs, :DocTestSetup, :(using SyntheticLogs); recursive=true)

makedocs(;
    modules=[SyntheticLogs],
    authors="Fedor Zolotarev",
    sitename="SyntheticLogs.jl",
    format=Documenter.HTML(;
        canonical="https://Dysthymiac.github.io/SyntheticLogs.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Dysthymiac/SyntheticLogs.jl",
    devbranch="main",
)
