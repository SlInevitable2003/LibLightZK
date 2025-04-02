#include "alt_bn128_init.cuh"

int main(int argc, char *argv[])
{
    libff::bigint<4> a("13d67a85c6da6c96c86a56c84da86c7da5c879ad");
    libff::bigint<4> b("de0a8c7d89a6dc79a6556ec86a56c85da7886a56adac85dafe6c5d876ec927d1");

    a.print();
    b.print();   

    return 0;
}