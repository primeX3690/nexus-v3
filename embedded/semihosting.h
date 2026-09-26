/* embedded/semihosting.h
 *
 * Direct ARM semihosting calls (ARM's own debug-monitor protocol, the
 * mechanism QEMU implements to let bare-metal code print to the host
 * console) - used instead of pulling in newlib's rdimon/libc layer,
 * since that layer's startup assumptions were what faulted on this board
 * (see README.md). This is a standard, minimal, from-scratch semihosting
 * shim: three operations (write a string, write one hex-printed int,
 * exit) - just the ARM "Angel" SVC convention, no OS, no libc.
 */
#ifndef NEXUS_SEMIHOSTING_H
#define NEXUS_SEMIHOSTING_H

#define SYS_WRITE0 0x04
#define SYS_EXIT   0x18

static inline int semihost_call(int reason, void *arg) {
    register int r0 __asm__("r0") = reason;
    register void *r1 __asm__("r1") = arg;
    __asm__ volatile ("bkpt #0xAB" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}

static inline void sh_puts(const char *s) {
    semihost_call(SYS_WRITE0, (void *)s);
}

/* Minimal itoa for non-negative and negative ints, no libc. */
static inline void sh_put_int(int v) {
    char buf[16];
    int i = 15;
    buf[i--] = '\0';
    int neg = v < 0;
    unsigned int u = neg ? (unsigned int)(-v) : (unsigned int)v;
    if (u == 0) buf[i--] = '0';
    while (u > 0) {
        buf[i--] = '0' + (u % 10);
        u /= 10;
    }
    if (neg) buf[i--] = '-';
    sh_puts(&buf[i + 1]);
}

static inline void sh_exit(void) {
    semihost_call(SYS_EXIT, (void *)0x20026); /* ADP_Stopped_ApplicationExit */
}

#endif
