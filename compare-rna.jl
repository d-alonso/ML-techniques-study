include("rna.jl")

resultadosRNA = []

arquitecturasRNA=[[15],[15,6],[5,4],[30],[30,15]]

println("--------------------------------------------------------------------------------------------------------------------------------------------------------")

for arquitectura in arquitecturasRNA
    resultadoRNA = trainRNA(arquitectura)
    push!(resultadosRNA, resultadoRNA)
end

mediastestRNA = [mean(result["test"]) for result in resultadosRNA]

println("La mejor arquitectura de RNA fue: ", arquitecturasRNA[findall(mediastestRNA .== maximum(mediastestRNA))[1]])
