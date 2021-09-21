#define CATCH_CONFIG_CONSOLE_WIDTH 300
#define CATCH_CONFIG_RUNNER
#include <Kokkos_Core.hpp>
#include <catch.hpp>

int main(int argc, char* argv[]) {
  Kokkos::InitArguments arguments;
  arguments.disable_warnings = true;
  Kokkos::initialize(arguments);

  std::ostringstream msg;
#if defined(__CUDACC__)
  Kokkos::Cuda::print_configuration(msg);
#endif
  std::cout << msg.str() << std::endl;
  int result = Catch::Session().run(argc, argv);
  Kokkos::finalize();
  return (result < 0xff ? result : 0xff);
}
