# SPDX-License-Identifier: GPL-2.0
srcs-$(CONFIG_PCIE_CADENCE) += cadence/
srcs-$(CONFIG_PCI_FTPCI100) += pci-ftpci100.c
srcs-$(CONFIG_PCI_HYPERV) += pci-hyperv.c
srcs-$(CONFIG_PCI_HYPERV_INTERFACE) += pci-hyperv-intf.c
srcs-$(CONFIG_PCI_MVEBU) += pci-mvebu.c
srcs-$(CONFIG_PCI_AARDVARK) += pci-aardvark.c
srcs-$(CONFIG_PCI_TEGRA) += pci-tegra.c
srcs-$(CONFIG_PCI_RCAR_GEN2) += pci-rcar-gen2.c
srcs-$(CONFIG_PCIE_RCAR_HOST) += pcie-rcar.c pcie-rcar-host.c
srcs-$(CONFIG_PCIE_RCAR_EP) += pcie-rcar.c pcie-rcar-ep.c
srcs-$(CONFIG_PCI_HOST_COMMON) += pci-host-common.c
srcs-$(CONFIG_PCI_HOST_GENERIC) += pci-host-generic.c
srcs-$(CONFIG_PCIE_XILINX) += pcie-xilinx.c
srcs-$(CONFIG_PCIE_XILINX_NWL) += pcie-xilinx-nwl.c
srcs-$(CONFIG_PCIE_XILINX_CPM) += pcie-xilinx-cpm.c
srcs-$(CONFIG_PCI_V3_SEMI) += pci-v3-semi.c
srcs-$(CONFIG_PCI_XGENE_MSI) += pci-xgene-msi.c
srcs-$(CONFIG_PCI_VERSATILE) += pci-versatile.c
srcs-$(CONFIG_PCIE_IPROC) += pcie-iproc.c
srcs-$(CONFIG_PCIE_IPROC_MSI) += pcie-iproc-msi.c
srcs-$(CONFIG_PCIE_IPROC_PLATFORM) += pcie-iproc-platform.c
srcs-$(CONFIG_PCIE_IPROC_BCMA) += pcie-iproc-bcma.c
srcs-$(CONFIG_PCIE_ALTERA) += pcie-altera.c
srcs-$(CONFIG_PCIE_ALTERA_MSI) += pcie-altera-msi.c
srcs-$(CONFIG_PCIE_ROCKCHIP) += pcie-rockchip.c
srcs-$(CONFIG_PCIE_ROCKCHIP_EP) += pcie-rockchip-ep.c
srcs-$(CONFIG_PCIE_ROCKCHIP_HOST) += pcie-rockchip-host.c
srcs-$(CONFIG_PCIE_MEDIATEK) += pcie-mediatek.c
srcs-$(CONFIG_PCIE_TANGO_SMP8759) += pcie-tango.c
srcs-$(CONFIG_VMD) += vmd.c
srcs-$(CONFIG_PCIE_BRCMSTB) += pcie-brcmstb.c
srcs-$(CONFIG_PCI_LOONGSON) += pci-loongson.c
# pcie-hisi.c quirks are needed even without CONFIG_PCIE_DW
subdirs-y				+= dwc
subdirs-y				+= mobiveil


# The following drivers are for devices that use the generic ACPI
# pci_root.c driver but don't support standard ECAM config access.
# They contain MCFG quirks to replace the generic ECAM accessors with
# device-specific ones that are shared with the DT driver.

# The ACPI driver is generic and should not require driver-specific
# config options to be enabled, so we always build these drivers on
# ARM64 and use internal ifdefs to only build the pieces we need
# depending on whether ACPI, the DT driver, or both are enabled.

ifdef CONFIG_PCI
srcs-$(CONFIG_ARM64) += pci-thunder-ecam.c
srcs-$(CONFIG_ARM64) += pci-thunder-pem.c
srcs-$(CONFIG_ARM64) += pci-xgene.c
endif
