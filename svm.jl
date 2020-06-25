
using ScikitLearn
using Statistics
using Flux: onehotbatch

@sk_import svm: SVC

using Random

include("funcionesUtiles.jl");
include("dataset.jl");

function trainAllSVM(gamma = "scale", c = 1.0, kern = "rbf", classes = [])
    results = []
    for class in [code(x) for x in classes]
        result = trainSVM(classes, gamma, c, kern, class)
        push!(results, result)
    end

    media_entrenamiento = mean(result["mean_train"] for result in results)
    desviacion_entrenamiento = std(result["std_train"] for result in results)
    media_test = mean(result["mean_test"] for result in results)
    desviacion_test = mean(result["std_test"] for result in results)

    println("       Resultados en promedio al separar todas las clases del resto con parámetros gamma: ", gamma, " y c: ",c," y kernel:", kern);
    println("           Entrenamiento: ", media_entrenamiento, " %, desviacion tipica: ", desviacion_entrenamiento);
    println("           Test:          ", media_test, " %, desviacion tipica: ", desviacion_test);
    println("------------------------------------------------------------------------------------------------------------------------------")

    return Dict("train" => media_entrenamiento, "test" => media_test, "model" => [gamma, c])
end;


function trainSVM(classes, gamma = "scale", c = 1.0, kern = "rbf", claseSeparar = code('l'))
    porcentajeTest = 0.3;
    numEjecuciones = 10;

    # Cargamos la base de datos de iris
    dataset = patterns(2000, classes)

    # Las entradas van a ser el tercer y cuarto atributo
    entradas = dataset[:, 2:end];
    # Cambiamos el tipo de la variable a un array de valores reales
    entradas = convert(Array{Float64}, entradas);

    # La salida deseada va a ser el quinto atributo
    salidasDeseadas = dataset[:,1];

    # El SVM es un clasificador de dos clases, asi que no puede separar las 3 clases
    # En lugar de ello, se usa una estrategia "uno contra todos", donde se separa cada clase del resto

    # En este ejemplo, vamos a separar la clase "virginica"
    # Por tanto, tenemos que generar salidas deseadas booleanas, una por patron

    # Las salidas deseadas se pueden generar de esta manera
    salidasDeseadasBool = (salidasDeseadas .== claseSeparar);
    # O, de forma alternativa, crear un "one hot encoding"
    #  Es decir, una matriz donde cada patron tiene varias salidas deseadas: una por clase
    #  La salida deseada de un patron es 1 para la clase a la que pertenece, y 0 para el resto
    clasesPosibles = unique(salidasDeseadas);
    # println(clasesPosibles)
    oneHotEncoding = Array(onehotbatch(salidasDeseadas, clasesPosibles));
    #     y despues seleccionamos aquella fila donde esta la clase que buscamos
    salidasDeseadasOneHot = vec(oneHotEncoding[clasesPosibles .== claseSeparar, :]);
    # Cualquiera de las dos formas deberia dar los mismos resultados:
    @assert (salidasDeseadasBool==salidasDeseadasOneHot);
    salidasDeseadas = salidasDeseadasBool;


    # Cuantos patrones tenemos
    numPatrones = size(entradas, 1);

    # Vamos a normalizar las entradas entre un maximo y un minimo
    # En este caso, cada patron esta en una fila (al trabajar con redes de neuronas esto es al reves)
    valoresNormalizacion = normalizarMaxMin!(entradas; patronesEnFilas=true);

    # Dado que es un problema de clasificacion, no hay que normalizar la salida
    # En un problema de regresion si que habria que normalizarla

    # Creamos un par de vectores vacios donde vamos a ir guardando los resultados de cada ejecucion
    precisionesEntrenamiento = Array{Float64,1}();
    precisionesTest = Array{Float64,1}();
    modelos = [];

    # Hacemos varias ejecuciones
    for numEjecucion in 1:numEjecuciones

        #println("Ejecucion ", numEjecucion);

        # Seleccionamos los patrones de entrenamiento y test: creamos indices
        # Esto es hacer un hold out
        (indicesPatronesEntrenamiento, indicesPatronesTest) = holdOut(numPatrones, porcentajeTest);

        # Creamos el SVM
        modelo = SVC(gamma=gamma, C=c, kernel=kern, degree=2);
        # Ajustamos el modelo, solo con los patrones de entrenamiento
        fit!(modelo, entradas[indicesPatronesEntrenamiento,:], salidasDeseadas[indicesPatronesEntrenamiento]);

        # Calculamos la clasificacion de los patrones de entrenamiento
        clasificacionEntrenamiento = predict(modelo, entradas[indicesPatronesEntrenamiento,:]);
        # Calculamos las distancias al hiperplano de esos mismos patrones
        distanciasHiperplano = decision_function(modelo, entradas[indicesPatronesEntrenamiento,:]);
        # Calculamos la precision en el conjunto de entrenamiento y la mostramos por pantalla
        precisionEntrenamiento = 100*mean(clasificacionEntrenamiento .== salidasDeseadas[indicesPatronesEntrenamiento]);
        #println("   Precision en el conjunto de entrenamiento: $precisionEntrenamiento %");

        # Calculamos la clasificacion de los patrones de test
        clasificacionTest = predict(modelo, entradas[indicesPatronesTest,:]);
        # Calculamos las distancias al hiperplano de esos mismos patrones
        distanciasHiperplano = decision_function(modelo, entradas[indicesPatronesTest,:]);
        # Calculamos la precision en el conjunto de test y la mostramos por pantalla
        precisionTest = 100*mean(clasificacionTest .== salidasDeseadas[indicesPatronesTest]);
        #println("   Precision en el conjunto de test: $precisionTest %");

        # Y guardamos esos valores de precision obtenidos en esta ejecucion
        push!(precisionesEntrenamiento, precisionEntrenamiento);
        push!(precisionesTest, precisionTest);
        push!(modelos, modelo);


    end;

    # println("       Resultados en promedio al separar la clase '", claseSeparar, "' del resto con parámetros gamma: ", gamma, " y c: ",c," y kernel:", kern);
    # println("           Entrenamiento: ", mean(precisionesEntrenamiento), " %, desviacion tipica: ", std(precisionesEntrenamiento));
    # println("           Test:          ", mean(precisionesTest), " %, desviacion tipica: ", std(precisionesTest));

    return Dict("mean_train" => mean(precisionesEntrenamiento),"mean_test" => mean(precisionesTest),
            "std_train" => std(precisionesEntrenamiento), "std_test" => std(precisionesTest), "modelo" => modelos)

end


########################################################################################################################
# Si se quiere tener mas informacion del modelo:

# # Para ver los campos del modelo
# println(keys(modelo));
#
# # Por ejemplo:
# modelo.C
# modelo.support_vectors_
# modelo.support_
