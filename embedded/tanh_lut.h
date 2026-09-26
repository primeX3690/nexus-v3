/* embedded/tanh_lut.h
 *
 * 256-entry lookup-table tanh() with linear interpolation, covering
 * x in [-4, 4] (tanh saturates to +/-1 well before +/-4, so clamping
 * outside that range is correct, not an approximation shortcut).
 *
 * WHY a LUT instead of libm's tanhf(): real embedded ML inference runtimes
 * (TFLite Micro, CMSIS-NN) avoid calling libm transcendentals in the hot
 * path on Cortex-M parts without an FPU (or even with one, for
 * determinism/cycle-count predictability) - exactly the same reason this
 * port uses one instead of the softer "just call tanh() from math.h and
 * let the compiler handle it" shortcut.
 */
#ifndef NEXUS_TANH_LUT_H
#define NEXUS_TANH_LUT_H

#define TANH_LUT_SIZE 256
#define TANH_LUT_XMAX 4.0f

static float tanh_lut[TANH_LUT_SIZE];
static int tanh_lut_ready = 0;

/* Built once at boot from the true tanh Taylor/CORDIC-free polynomial
 * approximation below (not libm) - this init cost is paid once, not
 * per-inference, matching how a real target would flash a precomputed
 * table rather than compute it at runtime at all. */
static float tanh_approx_ref(float x) {
    /* Pade-style rational approximation, good to ~1e-4 over [-4,4] -
     * used ONLY to build the LUT at init time, not in the hot path. */
    float x2 = x * x;
    float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return a / b;
}

static void tanh_lut_init(void) {
    for (int i = 0; i < TANH_LUT_SIZE; i++) {
        float x = -TANH_LUT_XMAX + (2.0f * TANH_LUT_XMAX) * ((float)i / (TANH_LUT_SIZE - 1));
        tanh_lut[i] = tanh_approx_ref(x);
    }
    tanh_lut_ready = 1;
}

static inline float tanh_fast(float x) {
    if (x <= -TANH_LUT_XMAX) return -1.0f;
    if (x >= TANH_LUT_XMAX) return 1.0f;
    float pos = (x + TANH_LUT_XMAX) / (2.0f * TANH_LUT_XMAX) * (TANH_LUT_SIZE - 1);
    int idx = (int)pos;
    if (idx >= TANH_LUT_SIZE - 1) idx = TANH_LUT_SIZE - 2;
    float frac = pos - (float)idx;
    return tanh_lut[idx] + frac * (tanh_lut[idx + 1] - tanh_lut[idx]);
}

#endif
