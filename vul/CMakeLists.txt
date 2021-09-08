

set(headers
        Flux.h
        Macros.h
        Grid.h
        Residual.h
        PerfectGas.h
        PhysicalFlux.h
        Solution.h
        Vulcan.h
        )
set(sources
        Grid.cpp
        )
add_library(core ${headers} ${sources})
add_library(vul::core ALIAS core)
target_include_directories(core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_link_libraries(core PUBLIC Kokkos::kokkos)

add_executable(ucgo main.cpp)
target_link_libraries(ucgo PUBLIC vul::core)
target_compile_definitions(ucgo PRIVATE -DASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../test/assets")
