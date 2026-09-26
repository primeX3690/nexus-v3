/* embedded/startup.c
 *
 * Minimal Cortex-M3 vector table + reset handler, written from scratch
 * rather than relying on newlib's generic hosted crt0 (which faulted on
 * this board model - a real finding, not swept under the rug: see
 * embedded/README.md "what didn't work first try"). This is standard
 * bare-metal practice: on Cortex-M, the first two words of flash ARE the
 * vector table (initial stack pointer, then Reset_Handler address), read
 * directly by hardware (and by QEMU's Cortex-M emulation) at boot -
 * nothing else runs before this.
 */
#include <stdint.h>

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;
extern int main(void);

void Reset_Handler(void);
static void Default_Handler(void) { while (1) {} }

void NMI_Handler(void)        __attribute__((weak, alias("Default_Handler")));
void HardFault_Handler(void)  __attribute__((weak, alias("Default_Handler")));
void MemManage_Handler(void)  __attribute__((weak, alias("Default_Handler")));
void BusFault_Handler(void)   __attribute__((weak, alias("Default_Handler")));
void UsageFault_Handler(void) __attribute__((weak, alias("Default_Handler")));
void SVC_Handler(void)        __attribute__((weak, alias("Default_Handler")));
void PendSV_Handler(void)     __attribute__((weak, alias("Default_Handler")));
void SysTick_Handler(void)    __attribute__((weak, alias("Default_Handler")));

__attribute__((section(".isr_vector")))
void (* const vector_table[])(void) = {
    (void (*)(void))&_estack,   /* initial stack pointer: top of SRAM */
    Reset_Handler,
    NMI_Handler,
    HardFault_Handler,
    MemManage_Handler,
    BusFault_Handler,
    UsageFault_Handler,
    0, 0, 0, 0,                 /* reserved */
    SVC_Handler,
    0, 0,                       /* reserved (debug monitor, reserved) */
    PendSV_Handler,
    SysTick_Handler,
};

void Reset_Handler(void) {
    uint32_t *src = &_sidata, *dst = &_sdata;
    while (dst < &_edata) *dst++ = *src++;      /* copy .data from flash to RAM */
    dst = &_sbss;
    while (dst < &_ebss) *dst++ = 0;            /* zero .bss */

    main();

    while (1) {}  /* main() returns via a semihosting exit; this is a safety net */
}
