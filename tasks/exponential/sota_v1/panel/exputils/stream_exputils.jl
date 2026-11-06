using ExponentialUtilities
using LinearAlgebra
using SparseArrays

const DEFAULT_TOL_MULTIPLIER = 1.0e-6

function metadata_number(header::String, name::String)
    match_result = match(Regex("\"" * name * "\"\\s*:\\s*([-+0-9.eE]+)"), header)
    match_result === nothing && error("missing protocol field: $name")
    return match_result.captures[1]
end

function read_header()
    read(stdin, 8) == codeunits("FAMEXP2\n") || error("bad protocol magic")
    json_length = Int(ltoh(read(stdin, UInt32)))
    header = String(read(stdin, json_length))
    n = parse(Int, metadata_number(header, "n"))
    nnz = parse(Int, metadata_number(header, "nnz"))
    requests = parse(Int, metadata_number(header, "requests"))
    t = parse(Float64, metadata_number(header, "t"))
    rtol = parse(Float64, metadata_number(header, "rtol"))

    colptr_u64 = Vector{UInt64}(undef, n + 1)
    rowval_u32 = Vector{UInt32}(undef, nnz)
    values = Vector{Float64}(undef, nnz)
    read!(stdin, colptr_u64)
    read!(stdin, rowval_u32)
    read!(stdin, values)

    # The wire matrix is full symmetric CSR. Its arrays are therefore also the
    # CSC representation of its transpose, which is the same mathematical L.
    colptr = Vector{Int}(undef, n + 1)
    rowval = Vector{Int}(undef, nnz)
    @inbounds for i in eachindex(colptr)
        colptr[i] = Int(colptr_u64[i]) + 1
    end
    @inbounds for i in eachindex(rowval)
        rowval[i] = Int(rowval_u32[i]) + 1
        values[i] = -values[i]
    end
    A = SparseMatrixCSC(n, n, colptr, rowval, values)
    return n, nnz, requests, t, rtol, A
end

function main()
    multiplier = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : DEFAULT_TOL_MULTIPLIER
    isfinite(multiplier) && multiplier > 0 || error("tolerance multiplier must be positive")

    write(stdout, "READY\n")
    flush(stdout)

    n, nonzeros, requests, t, rtol, A = read_header()
    B = Matrix{Float64}(undef, n, 2)
    y = Vector{Float64}(undef, n)

    if iszero(t)
        for _ in 1:requests
            read!(stdin, @view(B[:, 1]))
            read!(stdin, @view(B[:, 2]))
            copyto!(y, @view(B[:, 1]))
            write(stdout, y)
            flush(stdout)
        end
        return
    end

    operator_norm = opnorm(A, Inf)
    if iszero(operator_norm)
        for _ in 1:requests
            read!(stdin, @view(B[:, 1]))
            read!(stdin, @view(B[:, 2]))
            @. y = B[:, 1] + t * B[:, 2]
            write(stdout, y)
            flush(stdout)
        end
        return
    end

    if n == 1
        a = A[1, 1]
        exponential = exp(t * a)
        phi_integral = expm1(t * a) / a
        for _ in 1:requests
            read!(stdin, @view(B[:, 1]))
            read!(stdin, @view(B[:, 2]))
            y[1] = exponential * B[1, 1] + phi_integral * B[1, 2]
            write(stdout, y)
            flush(stdout)
        end
        return
    end

    # A wider basis avoids excessive internal timesteps when sparse matvecs are
    # large and the propagation interval is very stiff. The cutoff depends only
    # on public protocol quantities, never on case identity or right-hand sides.
    stiff_large = n >= 20_000 && t * operator_norm >= 10_000
    krylov_dimension = min(stiff_large ? 256 : 60, n)
    caches = ExponentialUtilities._phiv_timestep_caches(@view(B[:, 1]), krylov_dimension, 1)
    internal_tol = multiplier * rtol
    for _ in 1:requests
        read!(stdin, @view(B[:, 1]))
        read!(stdin, @view(B[:, 2]))
        phiv_timestep!(
            y, t, A, B; adaptive = true, tol = internal_tol,
            m = krylov_dimension, opnorm = operator_norm, ishermitian = true,
            NA = nonzeros, caches = caches
        )
        write(stdout, y)
        flush(stdout)
    end
end

main()
