/** @file
 *****************************************************************************
 * @author     This file is part of libff, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#include "alt_bn128_g1.cuh"
#include "alt_bn128_g2.cuh"
#include "alt_bn128_init.cuh"

namespace libff {

    bigint<alt_bn128_r_limbs> alt_bn128_modulus_r;
    bigint<alt_bn128_q_limbs> alt_bn128_modulus_q;

    alt_bn128_Fq alt_bn128_coeff_b;
    alt_bn128_Fq2 alt_bn128_twist;
    alt_bn128_Fq2 alt_bn128_twist_coeff_b;
    alt_bn128_Fq alt_bn128_twist_mul_by_b_c0;
    alt_bn128_Fq alt_bn128_twist_mul_by_b_c1;
    alt_bn128_Fq2 alt_bn128_twist_mul_by_q_X;
    alt_bn128_Fq2 alt_bn128_twist_mul_by_q_Y;

    bigint<alt_bn128_q_limbs> alt_bn128_ate_loop_count;
    bool alt_bn128_ate_is_loop_count_neg;
    bigint<12*alt_bn128_q_limbs> alt_bn128_final_exponent;
    bigint<alt_bn128_q_limbs> alt_bn128_final_exponent_z;
    bool alt_bn128_final_exponent_is_z_neg;

    void init_alt_bn128_params()
    {
        typedef bigint<alt_bn128_r_limbs> bigint_r;
        typedef bigint<alt_bn128_q_limbs> bigint_q;

        assert(sizeof(uint64_t) == 8 || sizeof(uint64_t) == 4); // Montgomery assumes this

        /* parameters for scalar field Fr */

        alt_bn128_modulus_r = bigint_r("30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001"); 
        assert(alt_bn128_Fr::modulus_is_valid());
        alt_bn128_Fr::Rsquared = bigint_r("216d0b17f4e44a58c49833d53bb808553fe3ab1e35c59e31bb8e645ae216da7");
        alt_bn128_Fr::Rcubed = bigint_r("cf8594b7fcc657c893cc664a19fcfed2a489cbe1cfbb6b85e94d8e1b4bf0040");
        alt_bn128_Fr::inv = 0xc2e1f593efffffff;
        alt_bn128_Fr::num_bits = 254;
        alt_bn128_Fr::euler = bigint_r("183227397098d014dc2822db40c0ac2e9419f4243cdcb848a1f0fac9f8000000");
        alt_bn128_Fr::s = 28;
        alt_bn128_Fr::t = bigint_r("30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f");
        alt_bn128_Fr::t_minus_1_over_2 = bigint_r("183227397098d014dc2822db40c0ac2e9419f4243cdcb848a1f0fac9f");
        alt_bn128_Fr::multiplicative_generator = alt_bn128_Fr("5");
        alt_bn128_Fr::root_of_unity = alt_bn128_Fr("2a3c09f0a58a7e8500e0a7eb8ef62abc402d111e41112ed49bd61b6e725b19f0");
        alt_bn128_Fr::nqr = alt_bn128_Fr("5");
        alt_bn128_Fr::nqr_to_t = alt_bn128_Fr("2a3c09f0a58a7e8500e0a7eb8ef62abc402d111e41112ed49bd61b6e725b19f0");

        /* parameters for base field Fq */

        alt_bn128_modulus_q = bigint_q("30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47");
        assert(alt_bn128_Fq::modulus_is_valid());
        alt_bn128_Fq::Rsquared = bigint_q("6d89f71cab8351f47ab1eff0a417ff6b5e71911d44501fbf32cfc5b538afa89");
        alt_bn128_Fq::Rcubed = bigint_q("20fd6e902d592544ef7f0b0c0ada0afb62f210e6a7283db6b1cd6dafda1530df");
        alt_bn128_Fq::inv = 0x87d20782e4866389;
        alt_bn128_Fq::num_bits = 254;
        alt_bn128_Fq::euler = bigint_q("183227397098d014dc2822db40c0ac2ecbc0b548b438e5469e10460b6c3e7ea3");
        alt_bn128_Fq::s = 1;
        alt_bn128_Fq::t = bigint_q("183227397098d014dc2822db40c0ac2ecbc0b548b438e5469e10460b6c3e7ea3");
        alt_bn128_Fq::t_minus_1_over_2 = bigint_q("c19139cb84c680a6e14116da060561765e05aa45a1c72a34f082305b61f3f51");
        alt_bn128_Fq::multiplicative_generator = alt_bn128_Fq("3");
        alt_bn128_Fq::root_of_unity = alt_bn128_Fq("30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd46");
        alt_bn128_Fq::nqr = alt_bn128_Fq("3");
        alt_bn128_Fq::nqr_to_t = alt_bn128_Fq("30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd46");

        /* parameters for twist field Fq2 */
        alt_bn128_Fq2::euler = bigint<2*alt_bn128_q_limbs>("492e25c3b1e5fce2ccd37be01a4690e5805c2a88b1bab031376fd2e1a6359c682344f4abd09216425280c4e36cb656e5301039684f560809daa2c5113aeb4d8");
        alt_bn128_Fq2::s = 4;
        alt_bn128_Fq2::t = bigint<2*alt_bn128_q_limbs>("925c4b8763cbf9c599a6f7c0348d21cb00b85511637560626edfa5c34c6b38d04689e957a1242c84a50189c6d96cadca602072d09eac1013b5458a2275d69b");
        alt_bn128_Fq2::t_minus_1_over_2 = bigint<2*alt_bn128_q_limbs>("492e25c3b1e5fce2ccd37be01a4690e5805c2a88b1bab031376fd2e1a6359c682344f4abd09216425280c4e36cb656e5301039684f560809daa2c5113aeb4d");
        alt_bn128_Fq2::non_residue = alt_bn128_Fq("30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd46");
        alt_bn128_Fq2::nqr = alt_bn128_Fq2(alt_bn128_Fq("2"),alt_bn128_Fq("1"));
        alt_bn128_Fq2::nqr_to_t = alt_bn128_Fq2(alt_bn128_Fq("b20dcb5704e326a0dd3ecd4f30515275398a41a4e1dc5d347cfbbedda71cf82"),alt_bn128_Fq("b1ffefd8885bf22252522c29527d19f05cfc50e9715370ab0f3a6ca462390c"));
        alt_bn128_Fq2::Frobenius_coeffs_c1[0] = alt_bn128_Fq("1");
        alt_bn128_Fq2::Frobenius_coeffs_c1[1] = alt_bn128_Fq("30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd46");

        // /* parameters for Fq6 */
        // alt_bn128_Fq6::non_residue = alt_bn128_Fq2(alt_bn128_Fq("9"),alt_bn128_Fq("1"));
        // alt_bn128_Fq6::Frobenius_coeffs_c1[0] = alt_bn128_Fq2(alt_bn128_Fq("1"),alt_bn128_Fq("0"));
        // alt_bn128_Fq6::Frobenius_coeffs_c1[1] = alt_bn128_Fq2(alt_bn128_Fq("21575463638280843010398324269430826099269044274347216827212613867836435027261"),alt_bn128_Fq("10307601595873709700152284273816112264069230130616436755625194854815875713954"));
        // alt_bn128_Fq6::Frobenius_coeffs_c1[2] = alt_bn128_Fq2(alt_bn128_Fq("21888242871839275220042445260109153167277707414472061641714758635765020556616"),alt_bn128_Fq("0"));
        // alt_bn128_Fq6::Frobenius_coeffs_c1[3] = alt_bn128_Fq2(alt_bn128_Fq("3772000881919853776433695186713858239009073593817195771773381919316419345261"),alt_bn128_Fq("2236595495967245188281701248203181795121068902605861227855261137820944008926"));
        // alt_bn128_Fq6::Frobenius_coeffs_c1[4] = alt_bn128_Fq2(alt_bn128_Fq("2203960485148121921418603742825762020974279258880205651966"),alt_bn128_Fq("0"));
        // alt_bn128_Fq6::Frobenius_coeffs_c1[5] = alt_bn128_Fq2(alt_bn128_Fq("18429021223477853657660792034369865839114504446431234726392080002137598044644"),alt_bn128_Fq("9344045779998320333812420223237981029506012124075525679208581902008406485703"));
        // alt_bn128_Fq6::Frobenius_coeffs_c2[0] = alt_bn128_Fq2(alt_bn128_Fq("1"),alt_bn128_Fq("0"));
        // alt_bn128_Fq6::Frobenius_coeffs_c2[1] = alt_bn128_Fq2(alt_bn128_Fq("2581911344467009335267311115468803099551665605076196740867805258568234346338"),alt_bn128_Fq("19937756971775647987995932169929341994314640652964949448313374472400716661030"));
        // alt_bn128_Fq6::Frobenius_coeffs_c2[2] = alt_bn128_Fq2(alt_bn128_Fq("2203960485148121921418603742825762020974279258880205651966"),alt_bn128_Fq("0"));
        // alt_bn128_Fq6::Frobenius_coeffs_c2[3] = alt_bn128_Fq2(alt_bn128_Fq("5324479202449903542726783395506214481928257762400643279780343368557297135718"),alt_bn128_Fq("16208900380737693084919495127334387981393726419856888799917914180988844123039"));
        // alt_bn128_Fq6::Frobenius_coeffs_c2[4] = alt_bn128_Fq2(alt_bn128_Fq("21888242871839275220042445260109153167277707414472061641714758635765020556616"),alt_bn128_Fq("0"));
        // alt_bn128_Fq6::Frobenius_coeffs_c2[5] = alt_bn128_Fq2(alt_bn128_Fq("13981852324922362344252311234282257507216387789820983642040889267519694726527"),alt_bn128_Fq("7629828391165209371577384193250820201684255241773809077146787135900891633097"));

        // /* parameters for Fq12 */

        // alt_bn128_Fq12::non_residue = alt_bn128_Fq2(alt_bn128_Fq("9"),alt_bn128_Fq("1"));
        // alt_bn128_Fq12::Frobenius_coeffs_c1[0]  = alt_bn128_Fq2(alt_bn128_Fq("1"),alt_bn128_Fq("0"));
        // alt_bn128_Fq12::Frobenius_coeffs_c1[1]  = alt_bn128_Fq2(alt_bn128_Fq("8376118865763821496583973867626364092589906065868298776909617916018768340080"),alt_bn128_Fq("16469823323077808223889137241176536799009286646108169935659301613961712198316"));
        // alt_bn128_Fq12::Frobenius_coeffs_c1[2]  = alt_bn128_Fq2(alt_bn128_Fq("21888242871839275220042445260109153167277707414472061641714758635765020556617"),alt_bn128_Fq("0"));
        // alt_bn128_Fq12::Frobenius_coeffs_c1[3]  = alt_bn128_Fq2(alt_bn128_Fq("11697423496358154304825782922584725312912383441159505038794027105778954184319"),alt_bn128_Fq("303847389135065887422783454877609941456349188919719272345083954437860409601"));
        // alt_bn128_Fq12::Frobenius_coeffs_c1[4]  = alt_bn128_Fq2(alt_bn128_Fq("21888242871839275220042445260109153167277707414472061641714758635765020556616"),alt_bn128_Fq("0"));
        // alt_bn128_Fq12::Frobenius_coeffs_c1[5]  = alt_bn128_Fq2(alt_bn128_Fq("3321304630594332808241809054958361220322477375291206261884409189760185844239"),alt_bn128_Fq("5722266937896532885780051958958348231143373700109372999374820235121374419868"));
        // alt_bn128_Fq12::Frobenius_coeffs_c1[6]  = alt_bn128_Fq2(alt_bn128_Fq("21888242871839275222246405745257275088696311157297823662689037894645226208582"),alt_bn128_Fq("0"));
        // alt_bn128_Fq12::Frobenius_coeffs_c1[7]  = alt_bn128_Fq2(alt_bn128_Fq("13512124006075453725662431877630910996106405091429524885779419978626457868503"),alt_bn128_Fq("5418419548761466998357268504080738289687024511189653727029736280683514010267"));
        // alt_bn128_Fq12::Frobenius_coeffs_c1[8]  = alt_bn128_Fq2(alt_bn128_Fq("2203960485148121921418603742825762020974279258880205651966"),alt_bn128_Fq("0"));
        // alt_bn128_Fq12::Frobenius_coeffs_c1[9]  = alt_bn128_Fq2(alt_bn128_Fq("10190819375481120917420622822672549775783927716138318623895010788866272024264"),alt_bn128_Fq("21584395482704209334823622290379665147239961968378104390343953940207365798982"));
        // alt_bn128_Fq12::Frobenius_coeffs_c1[10] = alt_bn128_Fq2(alt_bn128_Fq("2203960485148121921418603742825762020974279258880205651967"),alt_bn128_Fq("0"));
        // alt_bn128_Fq12::Frobenius_coeffs_c1[11] = alt_bn128_Fq2(alt_bn128_Fq("18566938241244942414004596690298913868373833782006617400804628704885040364344"),alt_bn128_Fq("16165975933942742336466353786298926857552937457188450663314217659523851788715"));

        // /* choice of short Weierstrass curve and its twist */

        alt_bn128_coeff_b = alt_bn128_Fq("3");
        alt_bn128_twist = alt_bn128_Fq2(alt_bn128_Fq("9"), alt_bn128_Fq("1"));
        alt_bn128_twist_coeff_b = alt_bn128_coeff_b * alt_bn128_twist.inverse();
        alt_bn128_twist_mul_by_b_c0 = alt_bn128_coeff_b * alt_bn128_Fq2::non_residue;
        alt_bn128_twist_mul_by_b_c1 = alt_bn128_coeff_b * alt_bn128_Fq2::non_residue;
        alt_bn128_twist_mul_by_q_X = alt_bn128_Fq2(alt_bn128_Fq("2fb347984f7911f74c0bec3cf559b143b78cc310c2c3330c99e39557176f553d"),
                                            alt_bn128_Fq("16c9e55061ebae204ba4cc8bd75a079432ae2a1d0b7c9dce1665d51c640fcba2"));
        alt_bn128_twist_mul_by_q_Y = alt_bn128_Fq2(alt_bn128_Fq("63cf305489af5dcdc5ec698b6e2f9b9dbaae0eda9c95998dc54014671a0135a"),
                                            alt_bn128_Fq("7c03cbcac41049a0704b5a7ec796f2b21807dc98fa25bd282d37f632623b0e3"));

        /* choice of group G1 */
        alt_bn128_G1::G1_zero = alt_bn128_G1(alt_bn128_Fq::zero(),
                                        alt_bn128_Fq::one(),
                                        alt_bn128_Fq::zero());
        alt_bn128_G1::G1_one = alt_bn128_G1(alt_bn128_Fq("1"),
                                        alt_bn128_Fq("2"),
                                        alt_bn128_Fq::one());
        alt_bn128_G1::wnaf_window_table.resize(0);
        alt_bn128_G1::wnaf_window_table.push_back(11);
        alt_bn128_G1::wnaf_window_table.push_back(24);
        alt_bn128_G1::wnaf_window_table.push_back(60);
        alt_bn128_G1::wnaf_window_table.push_back(127);

        alt_bn128_G1::fixed_base_exp_window_table.resize(0);
        // window 1 is unbeaten in [-inf, 4.99]
        alt_bn128_G1::fixed_base_exp_window_table.push_back(1);
        // window 2 is unbeaten in [4.99, 10.99]
        alt_bn128_G1::fixed_base_exp_window_table.push_back(5);
        // window 3 is unbeaten in [10.99, 32.29]
        alt_bn128_G1::fixed_base_exp_window_table.push_back(11);
        // window 4 is unbeaten in [32.29, 55.23]
        alt_bn128_G1::fixed_base_exp_window_table.push_back(32);
        // window 5 is unbeaten in [55.23, 162.03]
        alt_bn128_G1::fixed_base_exp_window_table.push_back(55);
        // window 6 is unbeaten in [162.03, 360.15]
        alt_bn128_G1::fixed_base_exp_window_table.push_back(162);
        // window 7 is unbeaten in [360.15, 815.44]
        alt_bn128_G1::fixed_base_exp_window_table.push_back(360);
        // window 8 is unbeaten in [815.44, 2373.07]
        alt_bn128_G1::fixed_base_exp_window_table.push_back(815);
        // window 9 is unbeaten in [2373.07, 6977.75]
        alt_bn128_G1::fixed_base_exp_window_table.push_back(2373);
        // window 10 is unbeaten in [6977.75, 7122.23]
        alt_bn128_G1::fixed_base_exp_window_table.push_back(6978);
        // window 11 is unbeaten in [7122.23, 57818.46]
        alt_bn128_G1::fixed_base_exp_window_table.push_back(7122);
        // window 12 is never the best
        alt_bn128_G1::fixed_base_exp_window_table.push_back(0);
        // window 13 is unbeaten in [57818.46, 169679.14]
        alt_bn128_G1::fixed_base_exp_window_table.push_back(57818);
        // window 14 is never the best
        alt_bn128_G1::fixed_base_exp_window_table.push_back(0);
        // window 15 is unbeaten in [169679.14, 439758.91]
        alt_bn128_G1::fixed_base_exp_window_table.push_back(169679);
        // window 16 is unbeaten in [439758.91, 936073.41]
        alt_bn128_G1::fixed_base_exp_window_table.push_back(439759);
        // window 17 is unbeaten in [936073.41, 4666554.74]
        alt_bn128_G1::fixed_base_exp_window_table.push_back(936073);
        // window 18 is never the best
        alt_bn128_G1::fixed_base_exp_window_table.push_back(0);
        // window 19 is unbeaten in [4666554.74, 7580404.42]
        alt_bn128_G1::fixed_base_exp_window_table.push_back(4666555);
        // window 20 is unbeaten in [7580404.42, 34552892.20]
        alt_bn128_G1::fixed_base_exp_window_table.push_back(7580404);
        // window 21 is never the best
        alt_bn128_G1::fixed_base_exp_window_table.push_back(0);
        // window 22 is unbeaten in [34552892.20, inf]
        alt_bn128_G1::fixed_base_exp_window_table.push_back(34552892);

        /* choice of group G2 */

        alt_bn128_G2::G2_zero = alt_bn128_G2(alt_bn128_Fq2::zero(),
                                        alt_bn128_Fq2::one(),
                                        alt_bn128_Fq2::zero());

        alt_bn128_G2::G2_one = alt_bn128_G2(alt_bn128_Fq2(alt_bn128_Fq("1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed"),
                                                    alt_bn128_Fq("198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2")),
                                        alt_bn128_Fq2(alt_bn128_Fq("12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa"),
                                                    alt_bn128_Fq("90689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b")),
                                        alt_bn128_Fq2::one());
        alt_bn128_G2::wnaf_window_table.resize(0);
        alt_bn128_G2::wnaf_window_table.push_back(5);
        alt_bn128_G2::wnaf_window_table.push_back(15);
        alt_bn128_G2::wnaf_window_table.push_back(39);
        alt_bn128_G2::wnaf_window_table.push_back(109);

        alt_bn128_G2::fixed_base_exp_window_table.resize(0);
        // window 1 is unbeaten in [-inf, 5.10]
        alt_bn128_G2::fixed_base_exp_window_table.push_back(1);
        // window 2 is unbeaten in [5.10, 10.43]
        alt_bn128_G2::fixed_base_exp_window_table.push_back(5);
        // window 3 is unbeaten in [10.43, 25.28]
        alt_bn128_G2::fixed_base_exp_window_table.push_back(10);
        // window 4 is unbeaten in [25.28, 59.00]
        alt_bn128_G2::fixed_base_exp_window_table.push_back(25);
        // window 5 is unbeaten in [59.00, 154.03]
        alt_bn128_G2::fixed_base_exp_window_table.push_back(59);
        // window 6 is unbeaten in [154.03, 334.25]
        alt_bn128_G2::fixed_base_exp_window_table.push_back(154);
        // window 7 is unbeaten in [334.25, 742.58]
        alt_bn128_G2::fixed_base_exp_window_table.push_back(334);
        // window 8 is unbeaten in [742.58, 2034.40]
        alt_bn128_G2::fixed_base_exp_window_table.push_back(743);
        // window 9 is unbeaten in [2034.40, 4987.56]
        alt_bn128_G2::fixed_base_exp_window_table.push_back(2034);
        // window 10 is unbeaten in [4987.56, 8888.27]
        alt_bn128_G2::fixed_base_exp_window_table.push_back(4988);
        // window 11 is unbeaten in [8888.27, 26271.13]
        alt_bn128_G2::fixed_base_exp_window_table.push_back(8888);
        // window 12 is unbeaten in [26271.13, 39768.20]
        alt_bn128_G2::fixed_base_exp_window_table.push_back(26271);
        // window 13 is unbeaten in [39768.20, 106275.75]
        alt_bn128_G2::fixed_base_exp_window_table.push_back(39768);
        // window 14 is unbeaten in [106275.75, 141703.40]
        alt_bn128_G2::fixed_base_exp_window_table.push_back(106276);
        // window 15 is unbeaten in [141703.40, 462422.97]
        alt_bn128_G2::fixed_base_exp_window_table.push_back(141703);
        // window 16 is unbeaten in [462422.97, 926871.84]
        alt_bn128_G2::fixed_base_exp_window_table.push_back(462423);
        // window 17 is unbeaten in [926871.84, 4873049.17]
        alt_bn128_G2::fixed_base_exp_window_table.push_back(926872);
        // window 18 is never the best
        alt_bn128_G2::fixed_base_exp_window_table.push_back(0);
        // window 19 is unbeaten in [4873049.17, 5706707.88]
        alt_bn128_G2::fixed_base_exp_window_table.push_back(4873049);
        // window 20 is unbeaten in [5706707.88, 31673814.95]
        alt_bn128_G2::fixed_base_exp_window_table.push_back(5706708);
        // window 21 is never the best
        alt_bn128_G2::fixed_base_exp_window_table.push_back(0);
        // window 22 is unbeaten in [31673814.95, inf]
        alt_bn128_G2::fixed_base_exp_window_table.push_back(31673815);

        /* pairing parameters */

        alt_bn128_ate_loop_count = bigint_q("19d797039be763ba8");
        alt_bn128_ate_is_loop_count_neg = false;
        alt_bn128_final_exponent = bigint<12*alt_bn128_q_limbs>("2f4b6dc97020fddadf107d20bc842d43bf6369b1ff6a1c71015f3f7be2e1e30a73bb94fec0daf15466b2383a5d3ec3d15ad524d8f70c54efee1bd8c3b21377e563a09a1b705887e72eceaddea3790364a61f676baaf977870e88d5c6c8fef0781361e443ae77f5b63a2a2264487f2940a8b1ddb3d15062cd0fb2015dfc6668449aed3cc48a82d0d602d268c7daab6a41294c0cc4ebe5664568dfc50e1648a45a4a1e3a5195846a3ed011a337a02088ec80e0ebae8755cfe107acf3aafb40494e406f804216bb10cf430b0f37856b42db8dc5514724ee93dfb10826f0dd4a0364b9580291d2cd65664814fde37ca80bb4ea44eacc5e641bbadf423f9a2cbf813b8d145da90029baee7ddadda71c7f3811c4105262945bba1668c3be69a3c230974d83561841d766f9c9d570bb7fbe04c7e8a6c3c760c0de81def35692da361102b6b9b2b918837fa97896e84abb40a4efb7e54523a486964b64ca86f120");
        alt_bn128_final_exponent_z = bigint_q("44e992b44a6909f1");
        alt_bn128_final_exponent_is_z_neg = false;

    }
} // libff
