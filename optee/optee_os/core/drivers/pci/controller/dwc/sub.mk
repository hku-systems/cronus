# SPDX-License-Identifier: GPL-2.0
srcs-$(CONFIG_PCIE_DW) += pcie-designware.c
srcs-$(CONFIG_PCIE_DW_HOST) += pcie-designware-host.c
srcs-$(CONFIG_PCIE_DW_EP) += pcie-designware-ep.c
srcs-$(CONFIG_PCIE_DW_PLAT) += pcie-designware-plat.c
srcs-$(CONFIG_PCI_DRA7XX) += pci-dra7xx.c
srcs-$(CONFIG_PCI_EXYNOS) += pci-exynos.c
srcs-$(CONFIG_PCI_IMX6) += pci-imx6.c
srcs-$(CONFIG_PCIE_SPEAR13XX) += pcie-spear13xx.c
srcs-$(CONFIG_PCI_KEYSTONE) += pci-keystone.c
srcs-$(CONFIG_PCI_LAYERSCAPE) += pci-layerscape.c
srcs-$(CONFIG_PCI_LAYERSCAPE_EP) += pci-layerscape-ep.c
srcs-$(CONFIG_PCIE_QCOM) += pcie-qcom.c
srcs-$(CONFIG_PCIE_ARMADA_8K) += pcie-armada8k.c
srcs-$(CONFIG_PCIE_ARTPEC6) += pcie-artpec6.c
srcs-$(CONFIG_PCIE_INTEL_GW) += pcie-intel-gw.c
srcs-$(CONFIG_PCIE_KIRIN) += pcie-kirin.c
srcs-$(CONFIG_PCIE_HISI_STB) += pcie-histb.c
srcs-$(CONFIG_PCI_MESON) += pci-meson.c
srcs-$(CONFIG_PCIE_TEGRA194) += pcie-tegra194.c
srcs-$(CONFIG_PCIE_UNIPHIER) += pcie-uniphier.c
srcs-$(CONFIG_PCIE_UNIPHIER_EP) += pcie-uniphier-ep.c

# The following drivers are for devices that use the generic ACPI
# pci_root.c driver but don't support standard ECAM config access.
# They contain MCFG quirks to replace the generic ECAM accessors with
# device-specific ones that are shared with the DT driver.

# The ACPI driver is generic and should not require driver-specific
# config options to be enabled, so we always build these drivers on
# ARM64 and use internal ifdefs to only build the pieces we need
# depending on whether ACPI, the DT driver, or both are enabled.

ifdef CONFIG_PCI
srcs-$(CONFIG_ARM64) += pcie-al.c
srcs-$(CONFIG_ARM64) += pcie-hisi.c
endif
