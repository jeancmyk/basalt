/* The two scalar F16C conversions, for a compiler that declares them but never generates them.
 *
 * MSVC turns _cvtss_sh and _cvtsh_ss into calls and no library answers them, so anything that uses them fails to
 * link. The vector forms do inline, and they are the same instruction with the same result: this builds the scalar
 * pair out of them. The rounding mode is a constant in the intrinsic, so the four cases are written out.
 *
 * Only the AVX2 and AVX-512 dot products reach these, and they run behind the run-time F16C probe.
 */
#if defined(_MSC_VER)
#include <immintrin.h>

unsigned short _cvtss_sh(float a, const int rounding) {
    const __m128 v = _mm_set_ss(a);
    switch (rounding & 3) {
        case 0:  return (unsigned short) _mm_cvtsi128_si32(_mm_cvtps_ph(v, 0));
        case 1:  return (unsigned short) _mm_cvtsi128_si32(_mm_cvtps_ph(v, 1));
        case 2:  return (unsigned short) _mm_cvtsi128_si32(_mm_cvtps_ph(v, 2));
        default: return (unsigned short) _mm_cvtsi128_si32(_mm_cvtps_ph(v, 3));
    }
}

float _cvtsh_ss(unsigned short a) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) a)));
}
#endif
