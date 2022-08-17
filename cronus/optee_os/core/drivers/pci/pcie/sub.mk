# SPDX-License-Identifier: GPL-2.0
#
# Makefile for PCI Express features and port driver

pcieportdrv-y			:= portdrv_core.c portdrv_pci.c err.c

# srcs-$(CONFIG_PCIEPORTBUS)	+= pcieportdrv.c
srcs-$(CONFIG_PCIEPORTBUS)	+= $(pcieportdrv-y)

srcs-$(CONFIG_PCIEASPM)		+= aspm.c
srcs-$(CONFIG_PCIEAER)		+= aer.c
srcs-$(CONFIG_PCIEAER_INJECT)	+= aer_inject.c
srcs-$(CONFIG_PCIE_PME)		+= pme.c
srcs-$(CONFIG_PCIE_DPC)		+= dpc.c
srcs-$(CONFIG_PCIE_PTM)		+= ptm.c
srcs-$(CONFIG_PCIE_BW)		+= bw_notification.c
srcs-$(CONFIG_PCIE_EDR)		+= edr.c
