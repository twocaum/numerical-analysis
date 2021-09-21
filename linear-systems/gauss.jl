
using LinearAlgebra

function gauss_jacobi(A::Matrix{Float64}, B::Vector{Float64}, x::Vector{Float64}, e=0.0001::Float64)
    n, m = size(A)
    
    if n != m
        throw(DomainError(A, "coefficients matrix should be squared"))
    end

    n_coef = diag(A)
    f = A - Diagonal(n_coef)
    i = 1
    while true
        x_0 = copy(x)
        x = (b - [dot(x_0, f[row, :]) for row=1:size(f)[1]]) ./ n_coef
        grad = x - x_0
        err = norm(grad)
        
        if err <= e
            break
        end
        i += 1
    end
end


function gauss_seidel(A::Matrix{Float64}, B::Vector{Float64}, x::Vector{Float64}, e=0.0001::Float64, maxit=typemax(Int64)::Int64)
    n, m = size(A)
    
    if n != m
        throw(DomainError(A, "coefficients matrix should be squared"))
    end

    n_coef = diag(A)
    f = A - Diagonal(n_coef)
    i = 1
    while i <= maxit
        x_0 = copy(x)
        for row=1:size(f)[1]
            x[row] = (b[row] - dot(x, f[row, :]))  / n_coef[row]
        end

        grad = x - x_0
        err = norm(grad)
        
        if err <= e
            break
        end
        
        i += 1
    end
end
