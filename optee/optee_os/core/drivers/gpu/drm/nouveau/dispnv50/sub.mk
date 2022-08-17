# SPDX-License-Identifier: MIT
srcs-y += disp.c
srcs-y += lut.c

srcs-y += core.c
srcs-y += core507d.c
srcs-y += core827d.c
srcs-y += core907d.c
srcs-y += core917d.c
srcs-y += corec37d.c
srcs-y += corec57d.c

srcs-$(CONFIG_DEBUG_FS) += crc.c
srcs-$(CONFIG_DEBUG_FS) += crc907d.c
srcs-$(CONFIG_DEBUG_FS) += crcc37d.c

srcs-y += dac507d.c
srcs-y += dac907d.c

srcs-y += pior507d.c

srcs-y += sor507d.c
srcs-y += sor907d.c
srcs-y += sorc37d.c

srcs-y += head.c
srcs-y += head507d.c
srcs-y += head827d.c
srcs-y += head907d.c
srcs-y += head917d.c
srcs-y += headc37d.c
srcs-y += headc57d.c

srcs-y += wimm.c
srcs-y += wimmc37b.c

srcs-y += wndw.c
srcs-y += wndwc37e.c
srcs-y += wndwc57e.c

srcs-y += base.c
srcs-y += base507c.c
srcs-y += base827c.c
srcs-y += base907c.c
srcs-y += base917c.c

srcs-y += curs.c
srcs-y += curs507a.c
srcs-y += curs907a.c
srcs-y += cursc37a.c

srcs-y += oimm.c
srcs-y += oimm507b.c

srcs-y += ovly.c
srcs-y += ovly507e.c
srcs-y += ovly827e.c
srcs-y += ovly907e.c
srcs-y += ovly917e.c
