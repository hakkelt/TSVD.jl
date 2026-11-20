using Test, TSVD, LinearAlgebra, SparseArrays

@testset "test tsvd with m = $m, n = $n, and p = $p" for
    (m, n, p) in ((10, 6, 0.8),
                  (6, 10, 0.8),
                  (10, 10, 0.8),
                  (100, 60, 0.1),
                  (60, 100, 0.1),
                  (100, 100, 0.1))

    mnp = round(Integer, m*n*p)

    for A in (randn(m, n),
              complex.(randn(m, n), randn(m, n)),
              sprandn(m, n, p),
              sparse(rand(1:m, mnp), rand(1:n, mnp), complex.(randn(mnp), randn(mnp)), m, n))

        Uf, sf, Vf = svd(Array(A))

        for k = 1:5
            U, s, V = TSVD.tsvd(A, k)
            @test norm(s - sf[1:k]) < sqrt(eps(real(eltype(A))))*mnp
            @test norm(abs.(U'Uf[:,1:k]) - I) < sqrt(eps(real(eltype(A))))*mnp
            @test norm(abs.(V'Vf[:,1:k]) - I) < sqrt(eps(real(eltype(A))))*mnp

            s, V = TSVD.tsvd2(A, k)
            # tmp = TSVD._teig(TSVD.AtA(A, randn(n)), k, debug = true)
            # @test norm(sqrt(reverse(tmp[1]))[1:k] - sf[1:k]) < sqrt(eps(real(eltype(A))))*mnp
            @test norm(s - sf[1:k]) < sqrt(eps(real(eltype(A))))*mnp
            @test norm(abs.(V'Vf[:,1:k]) - I) < sqrt(eps(real(eltype(A))))*mnp
        end
    end
end

@testset "Issue 9" begin
    data = rand(1:100, 50, 50)
    @test @inferred(TSVD.tsvd(data, 2))[2] ≈ svdvals(data)[1:2]
end

@testset "Issue 25. Degenerate basis" begin
  @test tsvd([1 0 ; 0 1], 1, initvec=[1.0; 0.0])[2] ≈ [1.0]
end

# Check that tsvd returns k singular triplets of A: values that match svdvals,
# orthonormal vectors, and small residuals.
function test_triplets(A, k)
    U, s, V = tsvd(A, k)
    sf = svdvals(Matrix(A))
    tol = sqrt(eps(real(eltype(A))))
    nrm = max(sf[1], 1)
    @test length(s) == k
    @test all(isfinite, s) && all(isfinite, U) && all(isfinite, V)
    @test norm(s - sf[1:k]) <= tol*nrm
    @test norm(U'U - I) <= tol
    @test norm(V'V - I) <= tol
    @test norm(A*V - U*Diagonal(s)) <= tol*nrm
    @test norm(A'U - V*Diagonal(s)) <= tol*nrm
end

@testset "Lanczos breakdown on rank-deficient matrices" begin
    @testset "zero matrix" begin
        test_triplets(zeros(4, 4), 1)
        test_triplets(zeros(ComplexF64, 5, 3), 2)
        test_triplets(zeros(Float32, 3, 6), 3)
    end
    @testset "rank below the number of values" begin
        test_triplets(ones(4, 4), 2)
        test_triplets(randn(6)*randn(5)', 2)
        test_triplets(complex.(randn(7), randn(7))*complex.(randn(5), randn(5))', 3)
        test_triplets(randn(Float32, 50, 2)*randn(Float32, 2, 40), 5)
    end
    @testset "Issue 48" begin
        test_triplets([202.0 0.0; 0.0 0.0], 1)
    end
    @testset "Issue 25. Rank one with a small dimension" begin
        for _ in 1:1000
            U, s, V = tsvd(randn(10)*randn(2)', 1)
            @test all(isfinite, s) && all(isfinite, U) && all(isfinite, V)
        end
        test_triplets(randn(10)*randn(2)', 2)
    end
end

@testset "all singular values (issue 49)" begin
    for (m, n) in ((4, 5), (5, 4), (6, 6), (1, 5), (5, 1))
        for A in (randn(m, n), complex.(randn(m, n), randn(m, n)))
            test_triplets(A, min(m, n))
        end
    end
    test_triplets(randn(4, 2)*randn(2, 5), 4)
    test_triplets(randn(5, 2)*randn(2, 4), 4)
    @test_throws ArgumentError tsvd(randn(4, 5), 5)
end
