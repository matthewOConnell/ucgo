

set(headers
        Macros.h
        Grid.h
        )
set(sources
        Grid.cpp
        )
add_library(core ${headers} ${sources})
add_library(vul::core ALIAS core)
target_include_directories(core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_link_libraries(core PUBLIC Kokkos::kokkos)
