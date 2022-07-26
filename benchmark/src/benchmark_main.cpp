#include <iostream>

#include "config.h"
#include "example_lib.h"



int main() {
    std::cout << "hello world" << std::endl;

    #if defined(USE_AVX)
        std::cout << "use AVX2" << std::endl;
    #elif defined(USE_SSE)
        std::cout << "use SSE" << std::endl;
    #else
        std::cout << "use arch" << std::endl;
    #endif

    exampleLib::test_print();

    return 0;
}