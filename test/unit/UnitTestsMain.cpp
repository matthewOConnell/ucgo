#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {
    doctest::Context context;

    Kokkos::InitArguments arguments;
    arguments.disable_warnings = true;
    Kokkos::initialize(arguments);

    std::ostringstream msg;
#if defined(__CUDACC__)
    Kokkos::Cuda::print_configuration(msg);
#endif
    std::cout << msg.str() << std::endl;

    int res = context.run(); // run

    if(context.shouldExit()) // important - query flags (and --exit) rely on the user doing this
        return res;          // propagate the result of the tests

    int client_stuff_return_code = 0;
    // your program - if the testing framework is integrated in your production code

    Kokkos::finalize();

    return res + client_stuff_return_code; // the result from doctest is propagated here as well
}
