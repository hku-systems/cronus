/* SPDX-License-Identifier: GPL-2.0 */
#ifndef __ASM_GENERIC_DELAY_H
#define __ASM_GENERIC_DELAY_H

/* Undefined functions to get compile-time errors */
extern void __bad_udelay(void);
extern void __bad_ndelay(void);

extern void __udelay(unsigned long usecs);
extern void __ndelay(unsigned long nsecs);
extern void __const_udelay(unsigned long xloops);
extern void __delay(unsigned long loops);

/*
 * The weird n/20000 thing suppresses a "comparison is always false due to
 * limited range of data type" warning with non-const 8-bit arguments.
 */

/* 0x10c7 is 2**32 / 1000000 (rounded up) */
#define udelay(n)							\
	({								\
		__udelay(n);					\
	})

/* 0x5 is 2**32 / 1000000000 (rounded up) */
#define ndelay(n)							\
	({								\
		__ndelay(n);					\
	})

#endif /* __ASM_GENERIC_DELAY_H */
