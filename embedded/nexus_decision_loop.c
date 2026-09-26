/* embedded/nexus_decision_loop.c
 *
 * Real bare-metal port of NEXUS's core per-tick decision loop:
 *
 *     sensor state --> PPO policy MLP forward pass --> proposed action
 *                  --> constitutional safety veto (priority-ordered)
 *                  --> final action
 *
 * matching brain/nexus_brain.py's documented pipeline order ("symbolic
 * safety layer runs LAST ... can veto/override the proposed action, but
 * never the other way around"). This is inference only (frozen weights
 * exported by export_weights.py from a real PPOAgent instance) - PPO
 * *training* (backprop, Adam, rollout buffers) is deliberately NOT ported;
 * that runs offline on the laptop, exactly like a real flight-software
 * split between ground-trained weights and onboard inference.
 *
 * Safety veto rules below mirror safety/constitutional_ai.py's four
 * VETO-capable rules in their exact priority order (100 > 95 > 90 > 85).
 * The fifth rule there (low_battery_return_to_base, priority 50) only
 * sets a "mode" fact and never vetoes an action, so it cannot change
 * final_action and is out of scope for this port - documented, not
 * silently dropped.
 *
 * Built for ARM Cortex-M3 (QEMU lm3s6965evb, same board used for
 * AscentGNC's embedded SIL port) via arm-none-eabi-gcc, run under QEMU's
 * instruction-accurate ARM core emulation. See embedded/README.md.
 *
 * MEASUREMENT METHOD: bracket_start()/bracket_end() are empty, non-inlined
 * functions with no other job than giving analyze_trace.py two addresses
 * to grep between in a QEMU -d exec,cpu instruction trace (the DWT cycle
 * counter is unimplemented on this QEMU board model - same finding as
 * AscentGNC's embedded port - so real instruction counting via
 * single-step trace is used instead of a cycle-counter read).
 */
#include "semihosting.h"
#include "tanh_lut.h"
#include "weights_state.h"

#define N_ITER 200

/* ---- PPO policy MLP forward pass (brain/rl/ppo_agent.py MLP.forward) ---- */
__attribute__((noinline))
static int policy_forward(const float state[NEXUS_STATE_DIM], float logits_out[NEXUS_N_ACTIONS]) {
    float hidden[NEXUS_HIDDEN_DIM];

    for (int h = 0; h < NEXUS_HIDDEN_DIM; h++) {
        float z = B1[h];
        for (int i = 0; i < NEXUS_STATE_DIM; i++) {
            z += state[i] * W1[i][h];
        }
        hidden[h] = tanh_fast(z);
    }

    int best_action = 0;
    float best_logit = -1e30f;
    for (int a = 0; a < NEXUS_N_ACTIONS; a++) {
        float z = B2[a];
        for (int h = 0; h < NEXUS_HIDDEN_DIM; h++) {
            z += hidden[h] * W2[h][a];
        }
        logits_out[a] = z;
        if (z > best_logit) {
            best_logit = z;
            best_action = a;
        }
    }
    return best_action;
}

/* ---- Constitutional safety veto (safety/constitutional_ai.py, priority order) ---- */
typedef struct {
    int final_action;
    int was_vetoed;
    const char *veto_reason;
} SafetyResult;

__attribute__((noinline))
static SafetyResult constitutional_check(int proposed_action) {
    SafetyResult r = { proposed_action, 0, "none" };

    if (FACT_BATTERY_PCT < 5.0f) {                                  /* priority 100 */
        r.final_action = HALT_ACTION; r.was_vetoed = 1; r.veto_reason = "critical_battery";
        return r;
    }
    if (FACT_MIN_OBSTACLE_DISTANCE < 0.3f) {                         /* priority 95 */
        r.final_action = HALT_ACTION; r.was_vetoed = 1; r.veto_reason = "collision_imminent";
        return r;
    }
    if (!FACT_IN_BOUNDS) {                                           /* priority 90 */
        r.final_action = HALT_ACTION; r.was_vetoed = 1; r.veto_reason = "out_of_bounds";
        return r;
    }
    if (FACT_STEPS_SINCE_LAST_COMM > FACT_COMM_TIMEOUT_STEPS) {      /* priority 85 */
        r.final_action = HALT_ACTION; r.was_vetoed = 1; r.veto_reason = "communication_lost";
        return r;
    }
    return r;
}

/* One full control-loop tick: perception-state-in, final-action-out. */
__attribute__((noinline))
static SafetyResult one_decision_tick(volatile float *sink, volatile int *proposed_out) {
    float logits[NEXUS_N_ACTIONS];
    int proposed = policy_forward(INPUT_STATE, logits);
    *sink = logits[0];  /* prevent the optimizer from discarding the whole computation */
    *proposed_out = proposed;
    return constitutional_check(proposed);
}

/* Markers analyze_trace.py locates in the disassembly to bound the
 * instruction-counting window - kept as real (non-inlined, non-optimized-
 * away) function calls so they show up as distinct addresses in the trace. */
__attribute__((noinline)) void bracket_start(void) { __asm__ volatile ("nop"); }
__attribute__((noinline)) void bracket_end(void)   { __asm__ volatile ("nop"); }

int main(void) {
    tanh_lut_init();
    volatile float sink = 0.0f;
    volatile int proposed_action = 0;
    SafetyResult result = { 0, 0, "none" };

    bracket_start();
    for (int i = 0; i < N_ITER; i++) {
        result = one_decision_tick(&sink, &proposed_action);
    }
    bracket_end();

    sh_puts("NEXUS embedded decision loop (Cortex-M3 / QEMU lm3s6965evb)\n");
    sh_puts("ran "); sh_put_int(N_ITER); sh_puts(" decision ticks\n");
    sh_puts("last tick: proposed="); sh_put_int(proposed_action);
    sh_puts(" final_action="); sh_put_int(result.final_action);
    sh_puts(" vetoed="); sh_put_int(result.was_vetoed);
    sh_puts(" reason="); sh_puts(result.veto_reason); sh_puts("\n");

    sh_exit();
    return 0;
}
