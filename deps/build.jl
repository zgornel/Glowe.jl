sourcedir = joinpath(dirname(@__FILE__), "src", "GloVe-c")
cd(sourcedir)
run(`make`)
