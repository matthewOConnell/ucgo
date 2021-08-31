add_subdirectory(doctest)

add_executable(UnitTests UnitTestsMain.cpp)
target_link_libraries(UnitTests PRIVATE doctest::doctest)
