

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
add_library(vul ${headers} ${sources})
add_library(vul::vul ALIAS vul)
target_include_directories(vul INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_link_libraries(vul PUBLIC Kokkos::kokkos)

add_executable(ucgo main.cpp)
target_link_libraries(ucgo PUBLIC vul::vul)
target_compile_definitions(ucgo PRIVATE -DASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../test/assets")
