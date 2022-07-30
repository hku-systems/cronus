/*
 * Copyright (c) 2017, Linaro Limited
 * SPDX-License-Identifier: BSD-2-Clause
 */

#ifndef COMMON_H
#define COMMON_H

/*
 * Misc. output defines
 */

#define OUTPUT_APP_PREFIX "Benchmark"

#ifdef DEBUG
#define DBG(fmt, args...) printf("[" OUTPUT_APP_PREFIX \
	"] DEBUG: %s:%d:%s(): " fmt "\n", __FILE__, __LINE__, __func__, ##args)
#else
#define DBG(fmt, args...)
#endif

#define INFO(fmt, args...) printf("[" OUTPUT_APP_PREFIX \
					"] INFO: " fmt "\n", ##args)
#define ERROR(fmt, args...) fprintf(stderr, "[" OUTPUT_APP_PREFIX \
					"] ERROR: " fmt "\n", ##args)

#define ERROR_EXIT(fmt, args...) do {		\
					ERROR(fmt, ##args);		\
					exit(EXIT_FAILURE);		\
			} while(0)

#define ERROR_RETURN_FALSE(fmt, args...) do {		\
					ERROR(fmt, ##args);		\
					return false;			\
			} while(0)

#define ERROR_GOTO(label, fmt, args...) do {		\
					ERROR(fmt, ##args);		\
					goto label;			\
			} while(0)




#define STAT_AMOUNT 				5
#define TSFILE_NAME_SUFFIX			".ts"
#define YAML_IMPLICIT				1
/*
 * Ringbuffer return codes
 */
#define RING_SUCCESS	0
#define RING_BADPARM	-1
#define RING_NODATA		-2

#define LIBTEEC_NAME	"libteec.so"
#endif /* COMMON.H */

