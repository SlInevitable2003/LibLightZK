#include "groth16.cuh"

int main(int argc, char *argv[])
{
    libff::init_alt_bn128_params();
    Groth16_BN254 protocol(input_scale::MatMul_p_100);
    protocol.Setup(12);
    protocol.Prove();
}