using SyntheticLogs
using Test
using Aqua
using JET

@testset "SyntheticLogs.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(SyntheticLogs)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(SyntheticLogs; target_defined_modules = true)
    end
    # Write your tests here.
end
