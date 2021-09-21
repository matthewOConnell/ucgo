add_subdirectory(catch2)


set(test_sources
        ArrayTests.cpp
        GridTests.cpp
        GasVariableTests.cpp
        LDFSSFluxTests.cpp
        PointTests.cpp
        CompressedRowGraph.cpp
        )

add_executable(UnitTests UnitTestsMain.cpp ${test_sources})
target_link_libraries(UnitTests PRIVATE vul::vul Catch2::Catch)
target_compile_definitions(UnitTests PRIVATE -DASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../assets")
