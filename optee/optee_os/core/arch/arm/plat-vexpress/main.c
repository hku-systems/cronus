// SPDX-License-Identifier: BSD-2-Clause
/*
 * Copyright (c) 2016-2020, Linaro Limited
 * Copyright (c) 2014, STMicroelectronics International N.V.
 */

#include <arm.h>
#include <console.h>
#include <drivers/gic.h>
#include <drivers/pl011.h>
#include <drivers/tzc400.h>
#include <initcall.h>
#include <keep.h>
#include <kernel/boot.h>
#include <kernel/interrupt.h>
#include <kernel/misc.h>
#include <kernel/panic.h>
#include <kernel/tee_time.h>
#include <mm/core_memprot.h>
#include <mm/core_mmu.h>
#include <platform_config.h>
#include <sm/psci.h>
#include <stdint.h>
#include <string.h>
#include <trace.h>

static struct gic_data gic_data __nex_bss;
static struct pl011_data console_data __nex_bss;

register_phys_mem_pgdir(MEM_AREA_IO_SEC, CONSOLE_UART_BASE, PL011_REG_SIZE);
#if defined(PLATFORM_FLAVOR_fvp)
register_phys_mem(MEM_AREA_RAM_SEC, TZCDRAM_BASE, TZCDRAM_SIZE);
#endif
#if defined(PLATFORM_FLAVOR_qemu_virt)
register_phys_mem_pgdir(MEM_AREA_IO_SEC, SECRAM_BASE, SECRAM_COHERENT_SIZE);
#endif
#ifdef DRAM0_BASE
register_ddr(DRAM0_BASE, DRAM0_SIZE);
#endif
#ifdef DRAM1_BASE
register_ddr(DRAM1_BASE, DRAM1_SIZE);
#endif

#ifdef GIC_BASE

register_phys_mem_pgdir(MEM_AREA_IO_SEC, GICD_BASE, GIC_DIST_REG_SIZE);
register_phys_mem_pgdir(MEM_AREA_IO_SEC, GICC_BASE, GIC_DIST_REG_SIZE);

void main_init_gic(void)
{
	vaddr_t gicc_base;
	vaddr_t gicd_base;

	gicc_base = (vaddr_t)phys_to_virt(GIC_BASE + GICC_OFFSET,
					  MEM_AREA_IO_SEC);
	gicd_base = (vaddr_t)phys_to_virt(GIC_BASE + GICD_OFFSET,
					  MEM_AREA_IO_SEC);
	if (!gicc_base || !gicd_base)
		panic();

#if defined(CFG_WITH_ARM_TRUSTED_FW)
	/* On ARMv8, GIC configuration is initialized in ARM-TF */
	gic_init_base_addr(&gic_data, gicc_base, gicd_base);
#else
	/* Initialize GIC */
	gic_init(&gic_data, gicc_base, gicd_base);
#endif
	itr_init(&gic_data.chip);
}

#if !defined(CFG_WITH_ARM_TRUSTED_FW)
void main_secondary_init_gic(void)
{
	gic_cpu_init(&gic_data);
}
#endif

#endif

void itr_core_handler(void)
{
	gic_it_handle(&gic_data);
}

void console_init(void)
{
	pl011_init(&console_data, CONSOLE_UART_BASE, CONSOLE_UART_CLK_IN_HZ,
		   CONSOLE_BAUDRATE);
	register_serial_console(&console_data.chip);
}

#if defined(IT_CONSOLE_UART) && \
	!(defined(CFG_WITH_ARM_TRUSTED_FW) && defined(CFG_ARM_GICV3))
/*
 * This cannot be enabled with TF-A and GICv3 because TF-A then need to
 * assign the interrupt number of the UART to OP-TEE (S-EL1). Currently
 * there's no way of TF-A to know which interrupts that OP-TEE will serve.
 * If TF-A doesn't assign the interrupt we're enabling below to OP-TEE it
 * will hang in EL3 since the interrupt will just be delivered again and
 * again.
 */
static enum itr_return console_itr_cb(struct itr_handler *h __unused)
{
	struct serial_chip *cons = &console_data.chip;

	while (cons->ops->have_rx_data(cons)) {
		int ch __maybe_unused = cons->ops->getchar(cons);

		DMSG("cpu %zu: got 0x%x", get_core_pos(), ch);
	}
	return ITRR_HANDLED;
}

static struct itr_handler console_itr = {
	.it = IT_CONSOLE_UART,
	.flags = ITRF_TRIGGER_LEVEL,
	.handler = console_itr_cb,
};
DECLARE_KEEP_PAGER(console_itr);

static TEE_Result init_console_itr(void)
{
	itr_add(&console_itr);
	itr_enable(IT_CONSOLE_UART);
	return TEE_SUCCESS;
}
driver_init(init_console_itr);
#endif

extern uint32_t init_vta_driver();
extern uint32_t init_gen_pcie_driver();
extern uint32_t init_drm_driver();
extern uint32_t init_drm_nouveau_driver();
extern uint32_t init_workqueue();
extern uint32_t init_radix_tree();
extern uint32_t init_drm_ttm();
extern uint32_t init_tee_time();
extern int buses_init(void);
extern int devices_init(void);

uint32_t init_drm_nouveau_int() {
	uint32_t exceptions = thread_get_exceptions();
	thread_set_exceptions(exceptions & ~THREAD_EXCP_ALL);
	uint32_t ret = init_drm_nouveau_driver();
	thread_set_exceptions(exceptions);
	return ret;
}

// extern int init_edu_driver(void);
// driver_init(init_edu_driver);
driver_init(init_vta_driver);
driver_init(init_drm_nouveau_int);
driver_init(init_drm_ttm);
driver_init(init_drm_driver);
driver_init(init_gen_pcie_driver);
driver_init(buses_init);
driver_init(devices_init);
driver_init(init_workqueue);
driver_init(init_radix_tree);
driver_init(init_tee_time);

#define NOUVEAU_PRI_BASE    0x20000000
#define NOUVEAU_PRI_SIZE    0x01000000

#define NOUVEAU_BIOS_BASE   0x21000000
#define NOUVEAU_BIOS_SIZE   0x00080000

#define NOUVEAU_PCI_BASE	0x2eff0000
#define NOUVEAU_PCI_SIZE	0x00010000

#define NOUVEAU_PRAMIN_BASE 0x18010000000
#define NOUVEAU_PRAMIN_SIZE 0x00000f00000

#define NOUVEAU_TTM_BO_BASE 0x18000200000
#define NOUVEAU_TTM_BO_SIZE 0x00000010000

#define NOUVEAU_UNKNOWN_BASE 0x20bb0000
#define NOUVEAU_UNKNOWN_SIZE 0x00010000

// this is vta_base when it is enable
#define NOUVEAU_CHANNEL_BASE 0x18000000000
#define NOUVEAU_CHANNEL_SIZE 0x10000
#define VTA_MEMORY_BASE 0x18000000000
#define VTA_MEMORY_SIZE 0x8000000

#define NOUVEAU2_PRAMIN_BASE 0x18050000000

#define NOUVEAU2_TTM_BO_BASE 0x18040200000

#define NOUVEAU2_CHANNEL_BASE 0x18040000000

register_phys_mem_pgdir(MEM_AREA_IO_SEC, 0x10000000000, 0x10000000);
register_phys_mem_pgdir(MEM_AREA_IO_SEC, NOUVEAU_PRI_BASE, NOUVEAU_PRI_SIZE);
register_phys_mem_pgdir(MEM_AREA_IO_SEC, NOUVEAU_PCI_BASE, NOUVEAU_PCI_SIZE);
register_phys_mem_pgdir(MEM_AREA_IO_SEC, NOUVEAU_BIOS_BASE, NOUVEAU_BIOS_SIZE);
register_phys_mem_pgdir(MEM_AREA_IO_SEC, NOUVEAU_PRAMIN_BASE, NOUVEAU_PRAMIN_SIZE);
register_phys_mem_pgdir(MEM_AREA_IO_SEC, NOUVEAU_TTM_BO_BASE, NOUVEAU_TTM_BO_SIZE);
register_phys_mem_pgdir(MEM_AREA_IO_SEC, VTA_MEMORY_BASE, VTA_MEMORY_SIZE);
register_phys_mem_pgdir(MEM_AREA_IO_SEC, NOUVEAU2_PRAMIN_BASE, NOUVEAU_PRAMIN_SIZE);
register_phys_mem_pgdir(MEM_AREA_IO_SEC, NOUVEAU2_TTM_BO_BASE, NOUVEAU_TTM_BO_SIZE);
register_phys_mem_pgdir(MEM_AREA_IO_SEC, NOUVEAU2_CHANNEL_BASE, NOUVEAU_CHANNEL_SIZE);

#define VRAM_BASE 	((long)1 << 32)
#define VHEAP_BASE 	(((long)1 << 32) + ((long)1 << 29))
#define VHEAP_RAM_SIZE 	((long)1 << 29)

#define EXTENDED_TA_RAM_SIZE (((long)1 << 27) - ((long)16 << 20))
#define VRAM_RAM_SIZE (((long)1 << 29))

#define TA_RAM_SIZE (((long)1 << 30))

register_phys_mem_pgdir(MEM_AREA_RAM_SEC, VRAM_BASE,  VRAM_RAM_SIZE);
register_phys_mem_pgdir(MEM_AREA_IO_SEC,  VHEAP_BASE, VHEAP_RAM_SIZE);
register_phys_mem(MEM_AREA_TA_RAM, VRAM_BASE + VRAM_RAM_SIZE + VHEAP_RAM_SIZE, TA_RAM_SIZE);

#ifdef CFG_TZC400
register_phys_mem_pgdir(MEM_AREA_IO_SEC, TZC400_BASE, TZC400_REG_SIZE);

static TEE_Result init_tzc400(void)
{
	void *va;

	DMSG("Initializing TZC400");

	va = phys_to_virt(TZC400_BASE, MEM_AREA_IO_SEC);
	if (!va) {
		EMSG("TZC400 not mapped");
		panic();
	}

	tzc_init((vaddr_t)va);
	tzc_dump_state();

	return TEE_SUCCESS;
}

service_init(init_tzc400);
#endif /*CFG_TZC400*/

#if defined(PLATFORM_FLAVOR_qemu_virt)
static void release_secondary_early_hpen(size_t pos)
{
	struct mailbox {
		uint64_t ep;
		uint64_t hpen[];
	} *mailbox;

	if (cpu_mmu_enabled())
		mailbox = phys_to_virt(SECRAM_BASE, MEM_AREA_IO_SEC);
	else
		mailbox = (void *)SECRAM_BASE;

	if (!mailbox)
		panic();

	mailbox->ep = TEE_LOAD_ADDR;
	dsb_ishst();
	mailbox->hpen[pos] = 1;
	dsb_ishst();
	sev();
}

int psci_cpu_on(uint32_t core_id, uint32_t entry, uint32_t context_id)
{
	size_t pos = get_core_pos_mpidr(core_id);
	static bool core_is_released[CFG_TEE_CORE_NB_CORE];

	if (!pos || pos >= CFG_TEE_CORE_NB_CORE)
		return PSCI_RET_INVALID_PARAMETERS;

	DMSG("core pos: %zu: ns_entry %#" PRIx32, pos, entry);

	if (core_is_released[pos]) {
		EMSG("core %zu already released", pos);
		return PSCI_RET_DENIED;
	}
	core_is_released[pos] = true;

	boot_set_core_ns_entry(pos, entry, context_id);
	release_secondary_early_hpen(pos);

	return PSCI_RET_SUCCESS;
}
#endif /*PLATFORM_FLAVOR_qemu_virt*/
