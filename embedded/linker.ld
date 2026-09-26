/* embedded/linker.ld
 * Memory map for QEMU's "lm3s6965evb" machine (TI Stellaris LM3S6965,
 * Cortex-M3): 256KB flash at 0x00000000, 64KB SRAM at 0x20000000 -
 * the real memory map of the physical part QEMU emulates, not a
 * fictional layout.
 */
MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 256K
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 64K
}

ENTRY(Reset_Handler)

SECTIONS
{
    .isr_vector : {
        KEEP(*(.isr_vector))
    } > FLASH

    .text : {
        *(.text*)
        *(.rodata*)
    } > FLASH

    _sidata = LOADADDR(.data);

    .data : {
        _sdata = .;
        *(.data*)
        _edata = .;
    } > RAM AT > FLASH

    .bss : {
        _sbss = .;
        *(.bss*)
        *(COMMON)
        _ebss = .;
    } > RAM

    ._user_heap_stack : {
        . = ALIGN(8);
        . = . + 0x1000;   /* 4KB stack */
        . = ALIGN(8);
    } > RAM

    _estack = ORIGIN(RAM) + LENGTH(RAM);
}
