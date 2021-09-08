add_subdirectory(doctest)


set(test_sources
        ArrayTests.cpp
        GridTests.cpp
        GasVariableTests.cpp
        LDFSSFluxTests.cpp
        PointTests.cpp
        )

add_executable(UnitTests UnitTestsMain.cpp ${test_sources})
target_link_libraries(UnitTests PRIVATE vul::core doctest::doctest)
target_compile_definitions(UnitTests PRIVATE -DASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../assets")
