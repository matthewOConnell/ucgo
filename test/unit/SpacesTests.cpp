#include <catch.hpp>
#include "../../vul/Macros.h"

TEST_CASE("Make sure host and device spaces are always different"){
  bool same_spaces = std::is_same<vul::Host, vul::Device>::value;
  REQUIRE(not same_spaces);
}