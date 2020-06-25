using CSV

#using ImageView
using Images

using Statistics
using Random

using RecursiveArrayTools

# código en el alfabeto (1-26) de cada caracter ASCII

function code(x::Char)::UInt8
    return lowercase(x) - 96
end

function letter(x::Int64)::Char
    return Char(x+96)
end

dataframe = CSV.read("emnist-letters-train.csv", delim=",")
arr = convert(Matrix{Int64}, dataframe)
dataframe = nothing; GC.gc(); #garbage collection
filas, cols = size(arr)

letras_bd = Dict()
for i = 1:26 # inicializar diccionario de las letras del 1 al 26
    letras_bd[i] = []
end

#introducir cada letra del dataset en el diccionario con su cód. como clave
for i = 1:filas
    push!(letras_bd[arr[i,1]], arr[i,1:end])
end

function rawpatterns(n_per_letter::Integer) # 768 características, 1 por píxel.
    output = zeros(Float64,n_per_letter*26,cols)
    for i=1:26
        tmp = arr[arr[:,1] .== i, :]
        output[1+(i-1)*n_per_letter:(i-1)*n_per_letter+n_per_letter,:] = tmp[1:n_per_letter,:]
    end
    return output[shuffle(1:end),:]
end

################### cambiar 'variables_*'' según aproximación ###################
function patterns(num_patterns::Integer, letters::Array{Char,1})

    patrones = Dict()
    for i in letters # inicializar diccionario de los patrones deseados
        patrones[i] = []
    end

    for j in letters
        for i = 1:div(num_patterns, size(letters)[1])
            imagen = reshape(letras_bd[code(j)][i][2:end], (28, 28))#formar la imagen de la lista de píxeles
            push!(patrones[j], variables_deep(j, imagen)) #extraer características
        end
    end

    i = 0
    output = nothing
    for (key, value) in patrones # juntar patrones
        if i==0
            output = hcat(value...)
        else
            output = hcat(value..., output)
        end
        i+=1
    end
    output = output[:, shuffle(1:end)] # orden aleatorio de patrones de entrada
    return output'
end;

######### Funciones de extracción de características

function variables_deep(letra, imagen)
    imagen = convert(Array{Float32}, imagen)
    imagen ./= 255.0 # normalizar: (pixel - 0) / (255 - 0)
    return [Int(code(letra)), imagen]
end;


function variables_entrada(letra, imagen)
    return [code(letra),
        mean(imagen[1:14, 1:14]), mean(imagen[1:14, 15:28]), mean(imagen[15:28, 1:14]), mean(imagen[15:28, 15:28]),
        std(imagen[1:14, 1:14]), std(imagen[1:14, 15:28]), std(imagen[15:28, 1:14]), std(imagen[15:28, 15:28])]
end;

function variables_entrada2(letra, imagen)
    return [
        code(letra),
        mean(imagen[1:7, 1:7]), mean(imagen[1:7, 7:14]), mean(imagen[1:7, 14:21]), mean(imagen[1:7, 21:28]),
        mean(imagen[7:14, 1:7]), mean(imagen[7:14, 7:14]), mean(imagen[7:14, 14:21]), mean(imagen[7:14, 21:28]),
        mean(imagen[14:21, 1:7]), mean(imagen[14:21, 7:14]), mean(imagen[14:21, 14:21]), mean(imagen[14:21, 21:28]),
        mean(imagen[21:28, 1:7]), mean(imagen[21:28, 7:14]), mean(imagen[21:28, 14:21]), mean(imagen[21:28, 21:28]),
        std(imagen[1:7, 1:7]), std(imagen[1:7, 7:14]), std(imagen[1:7, 14:21]), std(imagen[1:7, 21:28]),
        std(imagen[7:14, 1:7]), std(imagen[7:14, 7:14]), std(imagen[7:14, 14:21]), std(imagen[7:14, 21:28]),
        std(imagen[14:21, 1:7]), std(imagen[14:21, 7:14]), std(imagen[14:21, 14:21]), std(imagen[14:21, 21:28]),
        std(imagen[21:28, 1:7]), std(imagen[21:28, 7:14]), std(imagen[21:28, 14:21]), std(imagen[21:28, 21:28])
    ]
end;

function variables_entrada3(letra, imagen)
    return [
        code(letra),
        mean(imagen[1:7, 1:7]), mean(imagen[1:7, 7:14]), mean(imagen[1:7, 14:21]), mean(imagen[1:7, 21:28]),
        mean(imagen[7:14, 1:7]), mean(imagen[7:14, 7:14]), mean(imagen[7:14, 14:21]), mean(imagen[7:14, 21:28]),
        mean(imagen[14:21, 1:7]), mean(imagen[14:21, 7:14]), mean(imagen[14:21, 14:21]), mean(imagen[14:21, 21:28]),
        mean(imagen[21:28, 1:7]), mean(imagen[21:28, 7:14]), mean(imagen[21:28, 14:21]), mean(imagen[21:28, 21:28]),
        std(imagen[1:7, 1:7]), std(imagen[1:7, 7:14]), std(imagen[1:7, 14:21]), std(imagen[1:7, 21:28]),
        std(imagen[7:14, 1:7]), std(imagen[7:14, 7:14]), std(imagen[7:14, 14:21]), std(imagen[7:14, 21:28]),
        std(imagen[14:21, 1:7]), std(imagen[14:21, 7:14]), std(imagen[14:21, 14:21]), std(imagen[14:21, 21:28]),
        std(imagen[21:28, 1:7]), std(imagen[21:28, 7:14]), std(imagen[21:28, 14:21]), std(imagen[21:28, 21:28]),
        non0_pixels(imagen[1:7, 1:7]), non0_pixels(imagen[1:7, 7:14]), non0_pixels(imagen[1:7, 14:21]), non0_pixels(imagen[1:7, 21:28]),
        non0_pixels(imagen[7:14, 1:7]), non0_pixels(imagen[7:14, 7:14]), non0_pixels(imagen[7:14, 14:21]), non0_pixels(imagen[7:14, 21:28]),
        non0_pixels(imagen[14:21, 1:7]), non0_pixels(imagen[14:21, 7:14]), non0_pixels(imagen[14:21, 14:21]), non0_pixels(imagen[14:21, 21:28]),
        non0_pixels(imagen[21:28, 1:7]), non0_pixels(imagen[21:28, 7:14]), non0_pixels(imagen[21:28, 14:21]), non0_pixels(imagen[21:28, 21:28]),
        vertical_cut(imagen, 1), vertical_cut(imagen, 2),
        horizontal_cut(imagen, 1), horizontal_cut(imagen, 2)
    ]
end;

function non0_pixels(grid)
    counter=0
    for i in 1:7
        for j in 1:7
            if grid[i,j]!=0
                counter=counter+1
            end
        end
    end
    return counter
end;
# cortes horizontales y verticales: ver figura 1 en memoria.
function vertical_cut(imagen, cuadrante)
    counter=0
    start=cuadrante*14
    negro=true
    for j in cuadrante:(cuadrante+13)
        for i in 1:28
            if negro==true && imagen[i,j]!=0
                negro=false
                counter=counter+1
            elseif negro==false && imagen[i,j]==0
                negro=true
            end
        end
    end
    return counter
end;

function horizontal_cut(imagen, cuadrante)
    counter=0
    start=cuadrante*14
    negro=true
    for i in cuadrante:(cuadrante+13)
        for j in 1:28
            if negro==true && imagen[i,j]!=0
                negro=false
                counter=counter+1
            elseif negro==false && imagen[i,j]==0
                negro=true
            end
        end
    end
    return counter
end;
