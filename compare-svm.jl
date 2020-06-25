include("svm.jl")

letras_svm = ['l','m','o','p','q','u','v','w','z']

kernels = ["linear", "poly", "rbf", "sigmoid"]

arquitecturasSVM = [["auto", 1], ["auto", 3], ["auto", 0.5], [1, 5], [1, 0.1], [5, 1], [10,2], [4,2]]

for kernel in kernels
    resultadosSVM = []

    println("------------------------------------------------------------------------------------------------------------------------------")
    println("KERNEL ---> ", kernel)

    for arquitectura in arquitecturasSVM
        push!(resultadosSVM, trainAllSVM(arquitectura[1], arquitectura[2], kernel, letras_svm))
    end

    bestTest=0.0
    bestModel=[]
    for result in resultadosSVM
        if result["test"]>bestTest
            bestTest=result["test"]
            bestModel=result["model"]
        end
    end
    println("   El mejor SVM para el kernel ", kernel, " fue el de parámetros: gamma = ", bestModel[1], ", c = ", bestModel[2])
end
