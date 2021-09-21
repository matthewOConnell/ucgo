#include <cmath>
#include <catch.hpp>
#include <vul/Point.h>

TEST_CASE("Points can be accessed with x,y,z or pos[0:3]") {
  vul::Point<double> p(0, 1, 2);
  REQUIRE(p.x == 0);
  REQUIRE(p.y == 1);
  REQUIRE(p.z == 2);
  REQUIRE(p.pos[0] == 0);
  REQUIRE(p.pos[1] == 1);
  REQUIRE(p.pos[2] == 2);
}
TEST_CASE("Can take magnitude, and get normalized version") {
  vul::Point<double> p(0, 1, 2);
  REQUIRE(p.magnitude() == std::sqrt(5));
  REQUIRE(p.normalized().x == 0);
  REQUIRE(p.normalized().y == 1.0 / sqrt(5.0));
  REQUIRE(p.normalized().z == 2.0 / sqrt(5.0));
}
TEST_CASE("Can take dot product") {
  vul::Point<double> p = {0, 1, 2};
  REQUIRE(p.dot(p) == 5);
}

TEST_CASE("Can add / subtract product") {
  vul::Point<double> p = {0, 1, 2};

  p = p + p;
  REQUIRE(p.x == 0);
  REQUIRE(p.y == 2);
  REQUIRE(p.z == 4);
  p = p - p;
  REQUIRE(p.x == 0);
  REQUIRE(p.y == 0);
  REQUIRE(p.z == 0);
}

TEST_CASE("Can take cross product") {
  vul::Point<double> u = {1, 0, 0};
  vul::Point<double> v = {0, 1, 0};
  auto w               = u.cross(v);
  REQUIRE(w.x == 0);
  REQUIRE(w.y == 0);
  REQUIRE(w.z == 1);
}
