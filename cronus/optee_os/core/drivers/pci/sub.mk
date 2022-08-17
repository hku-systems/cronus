# SPDX-License-Identifier: GPL-2.0
#
# Makefile for the PCI bus specific drivers.

srcs-$(CONFIG_PCI)		+= access.c bus.c probe.c host-bridge.c \
				   remove.c pci.c pci-driver.c search.c \
				   pci-sysfs.c rom.c setup-res.c irq.c vpd.c \
				   setup-bus.c vc.c mmap.c setup-irq.c

subdirs-$(CONFIG_PCI) += pcie

ifdef CONFIG_PCI
srcs-$(CONFIG_PROC_FS)		+= proc.c
srcs-$(CONFIG_SYSFS)		+= slot.c
srcs-$(CONFIG_ACPI)		+= pci-acpi.c
endif

srcs-$(CONFIG_OF)		+= of.c
srcs-$(CONFIG_PCI_QUIRKS)	+= quirks.c

subdirs-$(CONFIG_HOTPLUG_PCI)	+= hotplug

srcs-$(CONFIG_PCI_MSI)		+= msi.c
srcs-$(CONFIG_PCI_ATS)		+= ats.c
srcs-$(CONFIG_PCI_IOV)		+= iov.c
srcs-$(CONFIG_PCI_BRIDGE_EMUL)	+= pci-bridge-emul.c
srcs-$(CONFIG_PCI_LABEL)		+= pci-label.c
srcs-$(CONFIG_X86_INTEL_MID)	+= pci-mid.c
srcs-$(CONFIG_PCI_SYSCALL)	+= syscall.c
srcs-$(CONFIG_PCI_STUB)		+= pci-stub.c
srcs-$(CONFIG_PCI_PF_STUB)	+= pci-pf-stub.c
srcs-$(CONFIG_PCI_ECAM)		+= ecam.c
srcs-$(CONFIG_PCI_P2PDMA)	+= p2pdma.c
srcs-$(CONFIG_XEN_PCIDEV_FRONTEND) += xen-pcifront.c

# Endpoint library must be initialized before its users
subdirs-$(CONFIG_PCI_ENDPOINT)	+= endpoint

subdirs-y += controller
subdirs-y += switch

ccflags-$(CONFIG_PCI_DEBUG) := -DDEBUG
ccflags-$(CONFIG_PCI) +=  -DCONFIG_64BIT

# cflags-y += -include core/include/kconfig.h

# -I./arch/x86/include -I./arch/x86/include/generated -I./include -I./arch/x86/include/uapi -I./arch/x86/include/generated/uapi -I./include/uapi -I./include/generated/uapi -include ./include/linux/kconfig.h -Iubuntu/include




