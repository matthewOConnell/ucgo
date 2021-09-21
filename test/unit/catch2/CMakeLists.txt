cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

add_library(Catch INTERFACE)
add_library(Catch2::Catch ALIAS Catch)
set(HEADER_FILES catch.hpp)

target_include_directories(Catch INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/catch2>
        )

install(FILES ${HEADER_FILES} DESTINATION include/catch2)

install(TARGETS Catch
        EXPORT Catch
        )
install(EXPORT Catch
        NAMESPACE Catch2:: DESTINATION cmake)
install(FILES Catch2Config.cmake
        DESTINATION cmake)

