/*
 * Copyright (C) Shinpei Kato
 *
 * University of California, Santa Cruz
 * Systems Research Lab.
 *
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __GDEV_API_H__
#define __GDEV_API_H__

#include "gdev_time.h" /* struct gdev_time */
#include "gdev_nvidia_def.h" /* struct gdev_kernel */
/* add also:
 * #include "gdev_amd_def.h"
 * #include "gdev_intel_def.h"
 */

/* Gdev handle members are not exposed to users. */
typedef struct gdev_handle* Ghandle;

/**
 * Gdev APIs:
 */
Ghandle gopen(int minor);
int gclose(Ghandle h);
uint64_t gmalloc(Ghandle h, uint64_t size);
uint64_t gfree(Ghandle h, uint64_t addr);
void *gmalloc_dma(Ghandle h, uint64_t size);
uint64_t gfree_dma(Ghandle h, void *buf);
void *gmap(Ghandle h, uint64_t addr, uint64_t size);
int gunmap(Ghandle h, void *buf);
int gmemcpy_to_device(Ghandle h, uint64_t dst_addr, const void *src_buf, uint64_t size);
int gmemcpy_to_device_async(Ghandle h, uint64_t dst_addr, const void *src_buf, uint64_t size, uint32_t *id);
int gmemcpy_user_to_device(Ghandle h, uint64_t dst_addr, const void *src_buf, uint64_t size);
int gmemcpy_user_to_device_async(Ghandle h, uint64_t dst_addr, const void *src_buf, uint64_t size, uint32_t *id);
int gmemcpy_from_device(Ghandle h, void *dst_buf, uint64_t src_addr, uint64_t size);
int gmemcpy_from_device_async(Ghandle h, void *dst_buf, uint64_t src_addr, uint64_t size, uint32_t *id);
int gmemcpy_user_from_device(Ghandle h, void *dst_buf, uint64_t src_addr, uint64_t size);
int gmemcpy_user_from_device_async(Ghandle h, void *dst_buf, uint64_t src_addr, uint64_t size, uint32_t *id);
int gmemcpy(Ghandle h, uint64_t dst_addr, uint64_t src_addr, uint64_t size);
int gmemcpy_async(Ghandle h, uint64_t dst_addr, uint64_t src_addr, uint64_t size, uint32_t *id);
int glaunch(Ghandle h, struct gdev_kernel *kernel, uint32_t *id);
int gsync(Ghandle h, uint32_t id, struct gdev_time *timeout);
int gbarrier(Ghandle h);
int gquery(Ghandle h, uint32_t type, uint64_t *result);
int gtune(Ghandle h, uint32_t type, uint32_t value);
int gshmget(Ghandle h, int key, uint64_t size, int flags);
uint64_t gshmat(Ghandle h, int id, uint64_t addr, int flags);
int gshmdt(Ghandle h, uint64_t addr);
int gshmctl(Ghandle h, int id, int cmd, void *buf);
uint64_t gref(Ghandle hmaster, uint64_t addr, uint64_t size, Ghandle hslave);
int gunref(Ghandle h, uint64_t addr);
uint64_t gphysget(Ghandle h, const void *p);
uint64_t gvirtget(Ghandle h, const void *p);
int gdevice_count(int* result);


/**
 * tuning types for Gdev resource management parameters.
 */
#define GDEV_TUNE_MEMCPY_PIPELINE_COUNT 1
#define GDEV_TUNE_MEMCPY_CHUNK_SIZE 2

/**
 * common queries:
 */
#define GDEV_QUERY_DEVICE_MEM_SIZE 1
#define GDEV_QUERY_DMA_MEM_SIZE 2
#define GDEV_QUERY_CHIPSET 3
#define GDEV_QUERY_BUS_TYPE 4
#define GDEV_QUERY_AGP_SIZE 5
#define GDEV_QUERY_PCI_VENDOR 6
#define GDEV_QUERY_PCI_DEVICE 7

/**
 * IPC commands:
 */
#define GDEV_IPC_STAT 1
#define GDEV_IPC_SET 2
#define GDEV_IPC_RMID 3

#define NV_CHIPSET_MASK 0x1f0

/* The following macro id is from the kernel tree 
 * linux/drivers/gpu/drm/nouveau/include/nvif/class.h
 */

#define NV_CHIPSET_NV50     0x50
#define NV_CHIPSET_FERMI    0xC0
#define NV_CHIPSET_KEPLER   0xE0
#define NV_CHIPSET_KEPLER_B 0xF0
#define NV_CHIPSET_MAXWELL  0x110
#define NV_CHIPSET_PASCAL   0x130
#define NV_CHIPSET_VOLTA    0x140
#define NV_CHIPSET_TURING   0x160
#define NV_CHIPSET_AMPERE   0x170


#define NV50_MEMORY_TO_MEMORY_FORMAT                                 0x00005039
#define FERMI_MEMORY_TO_MEMORY_FORMAT                                0x00009039
#define KEPLER_MEMORY_TO_MEMORY_FORMAT                               0x0000a040
#define KEPLER_B_MEMORY_TO_MEMORY_FORMAT                             0x0000a140

#define NV50_COMPUTE_A                                                 0x000050c0
#define FERMI_COMPUTE_A                                                0x000090c0
#define FERMI_COMPUTE_B                                                0x000091c0
#define KEPLER_COMPUTE_A                                               0x0000a0c0
#define KEPLER_COMPUTE_B                                               0x0000a1c0
#define MAXWELL_COMPUTE_A                                              0x0000b0c0
#define MAXWELL_COMPUTE_B                                              0x0000b1c0
#define PASCAL_COMPUTE_A                                               0x0000c0c0
#define PASCAL_COMPUTE_B                                               0x0000c1c0
#define VOLTA_COMPUTE_A                                                0x0000c3c0
#define TURING_COMPUTE_A                                               0x0000c5c0

#define CONCATENATE(arg1, arg2)   CONCATENATE1(arg1, arg2)
#define CONCATENATE1(arg1, arg2)  CONCATENATE2(arg1, arg2)
#define CONCATENATE2(arg1, arg2)  arg1##arg2

#define ASSIGN_CLASS_ELIF(v, suffix, chip_x, chip_y) \
	else if ((gdev->chipset & NV_CHIPSET_MASK) < NV_CHIPSET_##chip_y) v = chip_x##_##suffix;
#define ASSIGN_CLASS_CHIPSET(v, suffix, chip_x, chip_y, args...) \
	if ((gdev->chipset & NV_CHIPSET_MASK) < NV_CHIPSET_##chip_y) v = chip_x##_##suffix; \
	ASSIGN_IF_CHIPSET_ELIF(v, suffix, chip_y, args)

#define ASSIGN_IF_CHIPSET_ELIF_7(v, suffix, chip_x, chip_y, ...) \
	ASSIGN_CLASS_ELIF(v, suffix, chip_x, chip_y) \
	ASSIGN_IF_CHIPSET_ELIF_6(v, suffix, chip_y, __VA_ARGS__)

#define ASSIGN_IF_CHIPSET_ELIF_6(v, suffix, chip_default, chip_x, chip_y, ...) \
	ASSIGN_CLASS_ELIF(v, suffix, chip_x, chip_y) \
	ASSIGN_IF_CHIPSET_ELIF_5(v, suffix, chip_y)

#define ASSIGN_IF_CHIPSET_ELIF_5(v, suffix, chip_x, chip_y, ...) \
	ASSIGN_CLASS_ELIF(v, suffix, chip_x, chip_y) \
	ASSIGN_IF_CHIPSET_ELIF_4(v, suffix, chip_y, __VA_ARGS__)

#define ASSIGN_IF_CHIPSET_ELIF_4(v, suffix, chip_x, chip_y, ...) \
	ASSIGN_CLASS_ELIF(v, suffix, chip_x, chip_y) \
	ASSIGN_IF_CHIPSET_ELIF_3(v, suffix, chip_y, __VA_ARGS__)

#define ASSIGN_IF_CHIPSET_ELIF_3(v, suffix, chip_x, chip_y, ...) \
	ASSIGN_CLASS_ELIF(v, suffix, chip_x, chip_y) \
	ASSIGN_IF_CHIPSET_ELIF_2(v, suffix, chip_y, __VA_ARGS__)

#define ASSIGN_IF_CHIPSET_ELIF_2(v, suffix, chip_x, chip_y, ...) \
	ASSIGN_CLASS_ELIF(v, suffix, chip_x, chip_y) \
	ASSIGN_IF_CHIPSET_ELIF_1(v, suffix, chip_y, __VA_ARGS__)

#define ASSIGN_IF_CHIPSET_ELIF_1(v, suffix, chip_x, chip_y, ...) \
	ASSIGN_CLASS_ELIF(v, suffix, chip_x, chip_y) \
    ASSIGN_IF_CHIPSET_ELIF_0(v, suffix, chip_y, ignore)

#define ASSIGN_IF_CHIPSET_ELIF_0(v, suffix, chip_x, chip_y) \
    else v = chip_x##_##suffix; \

#define SIZEOF_NARG(...) SIZEOF_NARG_(__VA_ARGS__, SIZEOF_RSEQ_N())
#define SIZEOF_NARG_(...) SIZEOF_ARG_N(__VA_ARGS__)
#define SIZEOF_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
#define SIZEOF_RSEQ_N() 8, 7, 6, 5, 4, 3, 2, 1, 0

#define ASSIGN_IF_CHIPSET_ELIF_(N, v, suffix, x, ...) CONCATENATE(ASSIGN_IF_CHIPSET_ELIF_, N)(v, suffix, x, __VA_ARGS__)
#define ASSIGN_IF_CHIPSET_ELIF(v, suffix, x, ...) ASSIGN_IF_CHIPSET_ELIF_(SIZEOF_NARG(__VA_ARGS__), v, suffix, x, __VA_ARGS__)

#endif
