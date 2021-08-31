#pragma once
#include <string>
#include <stdexcept>

#define VUL_ASSERT( boolean_statement, message) { if(not (boolean_statement)) {throw std::logic_error(std::string("ASSERT_FAILED: ") + std::string(message) + " at file: " + std::string(__FILE__) + " function: " + std::string(__func__) + " line: " + std::to_string(__LINE__) + std::string("\n"));}}
