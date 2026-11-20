# Orthogonalize x against the vectors in Q. Two passes of modified Gram-Schmidt
# give orthogonality to working precision.
function _orthogonalize!(x, Q)
    for _ in 1:2, q in Q
        axpy!(-dot(q, x), q, x)
    end
    return x
end

# Called when the new Lanczos vector x has a small norm, so it may consist of
# rounding errors and components along Q that partial reorthogonalization has
# let through. Orthogonalize x fully against Q. If its norm is then at most tol,
# the Krylov subspace is invariant (breakdown): replace x by a random unit vector
# orthogonal to Q, or by zeros when Q already spans the whole space (room is
# false), and return a zero norm. Otherwise return the new norm; x is left
# unnormalized.
function _orthogonalize_or_restart!(x, Q, tol, room)
    _orthogonalize!(x, Q)
    nrm = norm(x)
    nrm > tol && return nrm
    if room
        randn!(x)
        _orthogonalize!(x, Q)
        rmul!(x, inv(norm(x)))
    else
        fill!(x, zero(eltype(x)))
    end
    return zero(nrm)
end

# A Lanczos coefficient below sqrt(eps)*norm(A) may be dominated by rounding
# errors; τ is eps times an estimate of norm(A).
_is_small(x, τ) = x <= sqrt(eps(typeof(τ)))*τ/eps(typeof(τ))

function _biLanczosIterations!(A, stepsize, αs, βs, U, V, μs, νs, maxνs, maxμs, τ, reorth_in, tolreorth, debug)

    m, n = size(A)
    # The bases cannot grow beyond the dimensions of A
    kmax = min(m, n)
    reorth_μ = reorth_in
    nReorth = 0
    nReorthVecs = 0

    T = eltype(eltype(U))
    Tr = real(T)

    iter = length(αs)

    u = U[iter + 1]
    v = V[iter]
    β = βs[iter]

    for j = iter .+ (1:stepsize)
        j > kmax && break

        # The v step
        vOld = v
        ## apply operator
        v = A'u
        axpy!(T(-β), vOld, v)
        α = norm(v)

        ## update norm(A) estimate. FixMe! Use tighter bounds, see Larsen's thesis page 33
        τ = max(τ, eps(Tr) * (α + β))
        debug && @show τ

        reorth_ν = false
        if _is_small(α, τ)
            ## possible breakdown; v is fully orthogonal to V afterwards
            debug && println("Small α: orthogonalize v")
            α = _orthogonalize_or_restart!(v, V, max(m, n)*τ, true)
            for i in 1:j - 1
                νs[i] = eps(Tr)
            end
            if j > 1
                push!(maxνs, eps(Tr))
            end
            push!(νs, 1)
            reorth_ν = true
        else
            ## run ω recurrence
            for i = 1:j - 1
                ν = βs[i]*μs[i + 1] + αs[i]*μs[i] - β*νs[i]
                ν = (ν + copysign(τ, ν))/α
                if abs(ν) > tolreorth
                    reorth_ν |= true
                end
                νs[i] = ν
            end
            if j > 1
                push!(maxνs, maximum(abs, νs))
            end
            push!(νs, 1)

            ## reorthogonalize if necessary
            if reorth_ν || reorth_μ
                debug && println("Reorth v")
                for i in 1:j - 1
                    axpy!(-dot(V[i], v), V[i], v)
                    νs[i] = eps(Tr)
                    nReorthVecs += 1
                end
                α = norm(v)
            end
        end

        ## update the result vectors
        push!(αs, α)
        iszero(α) || rmul!(v, inv(α))
        push!(V, v)

        # The u step
        uOld = u
        ## apply operator
        u = A*v
        axpy!(T(-α), uOld, u)
        β = norm(u)

        ## update norm(A) estimate. FixMe! Use tighter bounds, see Larsen's thesis page 33
        τ = max(τ, eps(Tr) * (α + β))
        debug && @show τ

        reorth_μ = false
        # When j == m, U already spans the whole space, so β is zero in exact
        # arithmetic.
        if _is_small(β, τ) || j == m
            ## possible breakdown; u is fully orthogonal to U afterwards
            debug && println("Small β: orthogonalize u")
            β = _orthogonalize_or_restart!(u, U, max(m, n)*τ, j < m)
            for i in 1:j
                μs[i] = eps(Tr)
            end
            push!(maxμs, eps(Tr))
            push!(μs, 1)
            reorth_μ = true
        else
            ## run ω recurrence
            for i = 1:j
                μ = αs[i]*νs[i] - α*μs[i]
                if i > 1
                    μ += βs[i - 1]*νs[i-1]
                end
                μ = (μ + copysign(τ, μ))/β
                if abs(μ) > tolreorth
                    reorth_μ |= true
                end
                μs[i] = μ
            end
            push!(maxμs, maximum(μs))
            push!(μs, 1)

            ## reorthogonalize if necessary
            if reorth_ν || reorth_μ
                debug && println("Reorth u")
                for i in 1:j
                    axpy!(-dot(U[i], u), U[i], u)
                    μs[i] = eps(Tr)
                    nReorthVecs += 1
                end
                β = norm(u)
                nReorth += 1
            end
        end

        ## update the result vectors
        push!(βs, β)
        iszero(β) || rmul!(u, inv(β))
        push!(U, u)
    end

    return reorth_μ
end

function biLanczos(A,
    nvals = 1;
    maxiter = 1000,
    initvec = convert(Vector{float(eltype(A))}, randn(size(A,1))),
    tolconv = sqrt(eps(real(eltype(initvec)))),
    tolreorth = sqrt(eps(real(eltype(initvec)))),
    stepsize = max(1, div(nvals, 10)),
    debug = false)

    Tv = eltype(initvec)
    Tr = real(Tv)

    m, n = size(A)
    kmax = min(m, n)
    if nvals > kmax
        throw(ArgumentError("cannot compute $nvals singular values of a $m×$n matrix"))
    end

    # I need to append βs with a zero at each iteration. Tt is much easier for type inference if it is a vector with the right element type
    z = zeros(Tr, 1)

    # initialize the αs, βs, U and V. Use result of first matvec to infer the correct types.
    # So the first iteration is run here, but slightly differently from the rest of the iterations
    nrmInit = norm(initvec)
    v = A'initvec
    rmul!(v, inv(nrmInit))
    α = norm(v)
    if iszero(α)
        # initvec is in the null space of A'. Continue from a random vector.
        _orthogonalize_or_restart!(v, typeof(v)[], zero(α), true)
    else
        if isone(α)
            # We need v to change when scaled with α.
            # Otherwise the basis generation might break down.
            α = nextfloat(α)
        end
        rmul!(v, inv(α))
    end
    V = fill(v, 1)
    αs = fill(α, 1)

    u = A*v
    uOld = similar(u)
    copyto!(uOld, initvec)
    rmul!(uOld, inv(nrmInit))
    axpy!(eltype(u)(-α), uOld, u)
    β = norm(u)

    # error estimate used in ω recurrence
    τ = eps(Tr)*(α + β)

    if _is_small(β, τ) || m == 1
        β = _orthogonalize_or_restart!(u, (uOld,), max(m, n)*τ, m > 1)
    else
        rmul!(u, inv(β))
    end
    U = typeof(u)[uOld, u]
    βs = fill(β, 1)

    μ = iszero(β) ? eps(Tr) : τ/β

    # Arrays for saving the estimates of the maximum angles between Lanczos vectors
    maxμs = Tr[]
    maxνs = Tr[]

    μs = Tr[μ, 1]
    νs = Tr[one(μ)]

    reorth_μ = _biLanczosIterations!(A, nvals - 1, αs, βs, U, V, μs, νs, maxμs, maxνs, τ, false, tolreorth, debug)

    # Iteration count
    iter = nvals

    # Save the estimates of the maximum angles between Lanczos vectors
    # append!(maxμs, maxμ)

    # Once the bases span the smaller dimension of A, the bidiagonal matrix
    # holds the singular values of A exactly.
    hasConv = length(αs) == kmax
    while !hasConv && iter <= maxiter

        reorth_μ = _biLanczosIterations!(A, stepsize, αs, βs, U, V, μs, νs, maxμs, maxνs, τ, reorth_μ, tolreorth, debug)
        iter += stepsize

        if length(αs) == kmax
            hasConv = true
            break
        end

        # This is more expensive than necessary because we only need the last components. However, LAPACK doesn't support this.
        UU, ss, VV = svd(Bidiagonal([αs;z], βs, :L))

        debug && @show βs[end]

        # Test for convergence. A Ritzvalue is considered converged if
        # either the last component of the corresponding vector is (relatively)
        # small or if the last component in βs is small (or both)
        # Use <= so that zero singular values of a rank-deficient A converge
        # once βs[end] is zero.
        if all(abs.(UU[end, 1:nvals])*βs[end] .<= tolconv*ss[1:nvals]) &&
           all(abs.(VV[end, 1:nvals])*βs[end] .<= tolconv*ss[1:nvals])
            hasConv = true
            break
        end

        τ = eps(eltype(ss))*ss[1]

        debug && @show iter
        debug && @show τ
    end
    if !hasConv
        error("no convergence")
    end

    if length(αs) == n < m
        # V spans the whole domain of A, so A*V = U*B̃ holds exactly, where B̃ is
        # the lower bidiagonal matrix with the extra row βs[end]*e_n'. Rotate that
        # row into a square upper bidiagonal matrix, which then has the singular
        # values of A, and rotate U to match.
        for j = 1:n
            c, s, r = LinearAlgebra.givensAlgorithm(αs[j], βs[j])
            uj, uj1 = U[j], U[j + 1]
            U[j], U[j + 1] = c*uj + s*uj1, c*uj1 - s*uj
            αs[j] = r
            if j < n
                βs[j] = s*αs[j + 1]
                αs[j + 1] = c*αs[j + 1]
            end
        end
        return U, Bidiagonal(αs, βs[1:end-1], :U), V, maxμs, maxνs
    end

    return U, Bidiagonal(αs, βs[1:end-1], :L), V, maxμs, maxνs
end

function _tsvd(A,
    nvals = 1;
    maxiter = 1000,
    # The initial vector is critical in determining the output type.
    # We use the result of A*initvec to detemine the storage type for
    # the Lanczos vectors. Hence, the user would need to either
    # have an appropriate multiplication method defined or supply
    # an appropriate initial vector.
    initvec = convert(Vector{float(eltype(A))}, randn(size(A,1))),
    tolconv = sqrt(eps(real(eltype(initvec)))),
    tolreorth = sqrt(eps(real(eltype(initvec)))),
    stepsize = max(1, div(nvals, 10)),
    debug = false)

    U, B, V, maxμs, maxνs = biLanczos(A,
        nvals;
        maxiter = maxiter,
        initvec = initvec,
        tolconv = tolconv,
        tolreorth = tolreorth,
        stepsize = stepsize,
        debug = debug)

    # Calculate the bidiagonal SVD
    smU, sms, smV = svd(B)

    # Create matrices from the Vectors of vectors and update U and V
    mU = hcat(U[1:end-1])
    mV = hcat(V)

    # Adapt the matrix type of the (LAPACK) SVD matrices to the type of mU and mV
    # E.g. if the vectors have been stored on a GPU device or on a distributed
    # system, this step will ensure that the result ends up where A/initvec is stored
    mUall = mU*adapt(typeof(initvec), smU)
    mVall = mV*adapt(typeof(initvec), smV)
    msall = adapt(typeof(initvec), sms)

    return mUall[:,1:nvals],
        sms[1:nvals],
        mVall[:,1:nvals],
        B,
        maxμs,
        maxνs
end

"""
    tsvd(A, nvals = 1; [maxiter, initvec, tolconv, tolreorth, debug])

Computes the truncated singular value decomposition (TSVD) by Lanczos bidiagonalization of the operator `A`. The Lanczos vectors are partially orthogonalized as described in

R. M. Larsen, *Lanczos bidiagonalization with partial reorthogonalization*, Department of Computer Science, Aarhus University, Technical report, DAIMI PB-357, September 1998.



# Positional arguments:

- `A`: Anything that supports the in place update operations


    mul!(y::AbstractVector, A, x::AbstractVector, α::Number, β::Number)

and

    mul!(y::AbstractVector, A::Adjoint, x::AbstractVector, α::Number, β::Number)

corresponding to the operations `y := α*op(A)*x + β*y` where `op` can be either the identity or the conjugate transpose of `A`. If the `initvec` argument is not supplied then it is furthermore required that `A` supports `eltype` and `size`.

- `nvals`: The number of singular values and vectors to compute, at most `minimum(size(A))`. Default is one (the largest).



# Keyword arguments:

- `maxiter`: The maximum number of iterations of the Lanczos bidiagonalization. Default is 1000, but usually much fewer iterations are needed.

- `initvec`: Initial `U` vector for the Lanczos procesdure. Default is a vector of Gaussian random variates. The `length` and `eltype` of the `initvec` will control the size and element types of the basis vectors in `U` and `V`.

- `tolconv`: Relative convergence criterion for the singular values. Default is `sqrt(eps(real(eltype(A))))`.

- `tolreorth`: Absolute tolerance for the inner product of the Lanczos vectors as measured by the ω recurrence. Default is `sqrt(eps(real(eltype(initvec))))`. `0.0` and `Inf` correspond to complete and no reorthogonalization respectively.

- `debug`: Boolean flag for printing debug information



# Output:

The output of the procesure it the truple tuple `(U, s, V)`

- `U`: `size(A, 1)` times `nvals` matrix of left singular vectors.
- `s`: Vector of length `nvals` of the singular values of `A`.
- `V`: `size(A, 2)` times `nvals` matrix of right singular vectors.



# Examples

```jldoctest
julia> A = matrixdepot("LPnetlib/lp_osa_30")
4350×104374 SparseArrays.SparseMatrixCSC{Float64, Int64} with 604488 stored entries:
⠙⠮⠷⠶⠽⠶⠽⠶⠮⠷⠮⠷⠶⠽⠶⠽⠶⠬⠷⠮⠷⠦⠽⠶⠽⠶⠽⠶⠮⠷⠮⠷⠶⠽⠶⠽⠶⠭⠷⠦

julia> U, s, V = tsvd(A, 5);

julia> round.(s, digits=7)
5-element Vector{Float64}:
 1365.8944098
 1033.2125634
  601.3524529
  554.107656
  506.0414587
```
"""
function tsvd(
    A,
    nvals = 1;
    maxiter = 1000,
    initvec = convert(Vector{float(eltype(A))}, randn(size(A,1))),
    tolconv = sqrt(eps(real(eltype(initvec)))),
    tolreorth = sqrt(eps(real(eltype(initvec)))),
    stepsize = max(1, div(nvals, 10)),
    debug = false)

    result = _tsvd(A, nvals, maxiter = maxiter, initvec = initvec, tolconv = tolconv,
            tolreorth = tolreorth, debug = debug)
    return (result[1], result[2], result[3])
end


### SVD by Lanczos on A'A

mutable struct AtA{T,S<:AbstractMatrix,V<:AbstractVecOrMat} <: AbstractMatrix{T}
    matrix::S
    vector::V
end
function AtA(A::AbstractMatrix, x::AbstractVecOrMat)
    y = A*x
    return AtA{eltype(y),typeof(A),typeof(y)}(A,y)
end

function size(A::AtA, i::Integer)
    if i < 1
        error("arraysize: dimension out of range")
    elseif i < 3
        return size(A.matrix, 2)
    else
        return 1
    end
end
size(A::AtA) = (size(A, 1), size(A, 2))

# Split Vector and Matrix to avoid ambiguity
function mul!(y::AbstractVector{T},
              A::AtA{T,S,V},
              x::AbstractVector{T},
              α::T = one(T),
              β::T = zero(T)) where {T<:Number,S,V}
    mul!(A.vector, A.matrix, x, one(T), zero(T))
    mul!(y, A.matrix', A.vector, α, β)
    return y
end
function mul!(y::AbstractMatrix{T},
              A::AtA{T,S,V},
              x::AbstractMatrix{T},
              α::T = one(T),
              β::T = zero(T)) where {T<:Number,S,V}
    mul!(A.vector, A.matrix, x, one(T), zero(T))
    mul!(y, A.matrix', A.vector, α, β)
    return y
end
(*)(A::AtA, x::AbstractVector) = mul!(similar(A.vector, size(x)), A, x)
(*)(A::AtA, x::AbstractMatrix) = mul!(similar(A.vector, size(x)), A, x)

function tsvd2(A,
    nvals = 1;
    maxiter = min(size(A)...),
    initvec = convert(Vector{eltype(A)}, randn(size(A,2))),
    tolconv = sqrt(eps(real(eltype(A)))),
    stepsize = max(1, div(nvals, 10)),
    debug = false)
    values, vectors, S, lanczosVecs = _teig(AtA(A, initvec), nvals, maxiter = maxiter,
        initvec = initvec, tolconv = tolconv, stepsize = stepsize, debug = debug)
    mV = hcat(lanczosVecs[1:end-1])*vectors
    return sqrt.(reverse(values)[1:nvals]), mV[:,end:-1:1][:,1:nvals]
end
