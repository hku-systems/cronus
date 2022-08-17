srcs-$(CFG_CDNS_UART) += cdns_uart.c
srcs-$(CFG_PL011) += pl011.c
srcs-$(CFG_TZC400) += tzc400.c
srcs-$(CFG_TZC380) += tzc380.c
srcs-$(CFG_GIC) += gic.c
srcs-$(CFG_PL061) += pl061_gpio.c
srcs-$(CFG_PL022) += pl022_spi.c
srcs-$(CFG_SP805_WDT) += sp805_wdt.c
srcs-$(CFG_8250_UART) += serial8250_uart.c
srcs-$(CFG_16550_UART) += ns16550.c
srcs-$(CFG_IMX_SNVS) += imx_snvs.c
srcs-$(CFG_IMX_UART) += imx_uart.c
srcs-$(CFG_IMX_I2C) += imx_i2c.c
srcs-$(CFG_IMX_LPUART) += imx_lpuart.c
srcs-$(CFG_IMX_WDOG) += imx_wdog.c
srcs-$(CFG_SPRD_UART) += sprd_uart.c
srcs-$(CFG_HI16XX_UART) += hi16xx_uart.c
srcs-$(CFG_HI16XX_RNG) += hi16xx_rng.c
srcs-$(CFG_SCIF) += scif.c
srcs-$(CFG_DRA7_RNG) += dra7_rng.c
srcs-$(CFG_STIH_UART) += stih_asc.c
srcs-$(CFG_ATMEL_UART) += atmel_uart.c
srcs-$(CFG_AMLOGIC_UART) += amlogic_uart.c
srcs-$(CFG_MVEBU_UART) += mvebu_uart.c
srcs-$(CFG_STM32_BSEC) += stm32_bsec.c
srcs-$(CFG_STM32_ETZPC) += stm32_etzpc.c
srcs-$(CFG_STM32_GPIO) += stm32_gpio.c
srcs-$(CFG_STM32_I2C) += stm32_i2c.c
srcs-$(CFG_STM32_RNG) += stm32_rng.c
srcs-$(CFG_STM32_UART) += stm32_uart.c
srcs-$(CFG_STPMIC1) += stpmic1.c
srcs-$(CFG_BCM_HWRNG) += bcm_hwrng.c
srcs-$(CFG_BCM_SOTP) += bcm_sotp.c
srcs-$(CFG_BCM_GPIO) += bcm_gpio.c
srcs-$(CFG_LS_I2C) += ls_i2c.c
srcs-$(CFG_LS_GPIO) += ls_gpio.c
srcs-$(CFG_LS_DSPI) += ls_dspi.c
srcs-$(CFG_IMX_RNGB) += imx_rngb.c

subdirs-y += crypto
subdirs-$(CFG_BNXT_FW) += bnxt
subdirs-$(CFG_SCMI_MSG_DRIVERS) += scmi-msg
subdirs-y += imx

subdirs-$(CONFIG_PCI) += pci

srcs-y += lib/logic_pio.c lib/bitmap.c lib/kstrtox.c lib/string.c lib/hexdump.c lib/devres.c 
srcs-y += lib/kobject.c lib/kobject_uevent.c lib/klist.c lib/cmdline.c lib/find_bit.c
srcs-y += lib/idr.c lib/radix-tree.c lib/ctype.c lib/rbtree.c lib/sort.c lib/list_sort.c
srcs-y += lib/xarray.c lib/uuid.c lib/refcount.c lib/kasprintf.c lib/hweight.c
srcs-y += lib/scatterlist.c lib/llist.c lib/genalloc.c
srcs-y += lib/pci_iomap.c
srcs-y += base/firmware_loader/main.c
subdirs-y += base/firmware_loader/builtin
# srcs-y += lib/vsprintf.c
# srcs-y +=  base/core.c base/bus.c base/dd.c base/driver.c base/class.c base/platform.c base/property.c
# srcs-y += base/swnode.c
srcs-y += base/platform.c base/core.c base/devres.c base/bus.c base/class.c base/devtmpfs.c
srcs-y += mm/util.c mm/dma-mapping.c mm/cma.c mm/cache.S
# mm/slab_common.c 
# mm/slub.c
srcs-y += kernel/resource.c kernel/cpu.c kernel/io_arm64.c
# kernel/time/jiffies.c
# srcs-y += fs/seq_file.c
srcs-y += of/address.c of/base.c of/property.c of/platform.c

srcs-y += edu.c

subdirs-$(CONFIG_DRM) += gpu/drm
subdirs-$(CONFIG_HAS_DMA) += dma
subdirs-$(CONFIG_HAS_DMA) += dma-buf

subdirs-y += vta