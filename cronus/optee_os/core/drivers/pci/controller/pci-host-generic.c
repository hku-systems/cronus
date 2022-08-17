// SPDX-License-Identifier: GPL-2.0
/*
 * Simple, generic PCI host controller driver targeting firmware-initialised
 * systems and virtual machines (e.g. the PCI emulation provided by kvmtool).
 *
 * Copyright (C) 2014 ARM Limited
 *
 * Author: Will Deacon <will.deacon@arm.com>
 */

#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/of.h>
#include <linux/of_platform.h>
#include <linux/pci-ecam.h>
#include <linux/platform_device.h>

static const struct pci_ecam_ops gen_pci_cfg_cam_bus_ops = {
	.bus_shift	= 16,
	.pci_ops	= {
		.map_bus	= pci_ecam_map_bus,
		.read		= pci_generic_config_read,
		.write		= pci_generic_config_write,
	}
};

static bool pci_dw_valid_device(struct pci_bus *bus, unsigned int devfn)
{
	struct pci_config_window *cfg = bus->sysdata;

	/*
	 * The Synopsys DesignWare PCIe controller in ECAM mode will not filter
	 * type 0 config TLPs sent to devices 1 and up on its downstream port,
	 * resulting in devices appearing multiple times on bus 0 unless we
	 * filter out those accesses here.
	 */
	if (bus->number == cfg->busr.start && PCI_SLOT(devfn) > 0)
		return false;

	return true;
}

static void __iomem *pci_dw_ecam_map_bus(struct pci_bus *bus,
					 unsigned int devfn, int where)
{
	if (!pci_dw_valid_device(bus, devfn))
		return NULL;

	return pci_ecam_map_bus(bus, devfn, where);
}

static const struct pci_ecam_ops pci_dw_ecam_bus_ops = {
	.bus_shift	= 20,
	.pci_ops	= {
		.map_bus	= pci_dw_ecam_map_bus,
		.read		= pci_generic_config_read,
		.write		= pci_generic_config_write,
	}
};

static const struct of_device_id gen_pci_of_match[] = {
	{ .compatible = "pci-host-cam-generic",
	  .data = &gen_pci_cfg_cam_bus_ops },

	{ .compatible = "pci-host-ecam-generic",
	  .data = &pci_generic_ecam_ops },

	{ .compatible = "marvell,armada8k-pcie-ecam",
	  .data = &pci_dw_ecam_bus_ops },

	{ .compatible = "socionext,synquacer-pcie-ecam",
	  .data = &pci_dw_ecam_bus_ops },

	{ .compatible = "snps,dw-pcie-ecam",
	  .data = &pci_dw_ecam_bus_ops },

	{ },
};
MODULE_DEVICE_TABLE(of, gen_pci_of_match);

static struct platform_driver gen_pci_driver = {
	.driver = {
		.name = "pci-host-generic",
		.of_match_table = gen_pci_of_match,
	},
	.probe = pci_host_common_probe,
	.remove = pci_host_common_remove,
};
// module_platform_driver(gen_pci_driver);

const u32 platform_prop_size_value[] = {cpu_to_be32(2)};

const u32 platform_prop_address_value[] = {cpu_to_be32(2)};

const struct property platform_properties[] = {
	{
		.name = "#address-cells",
		.length = sizeof(u32),
		.next = &platform_properties[1],
		.value = platform_prop_address_value
	},
	{
		.name = "#size-cells",
		.length = sizeof(u32),
		.next = NULL,
		.value = platform_prop_size_value
	}
};

struct device_node platform_device_node = {
	.name = "platform-controller",
	.full_name = "platform-controller-full",
	.properties = platform_properties,
	.parent = NULL
};

const u32 pci_prop_reg_values[] = {cpu_to_be32(0x100), 0, 0, cpu_to_be32(0x10000000)};

const u32 pci_prop_ranges_value[] = {
	cpu_to_be32(0x01000000), 0x00000000, 0x00000000, 0x00000000, cpu_to_be32(0x2eff0000), 0x00000000, cpu_to_be32(0x00010000), 
	cpu_to_be32(0x02000000), 0x00000000, cpu_to_be32(0x20000000), 0x00000000, cpu_to_be32(0x20000000), 0x00000000, cpu_to_be32(0x0eff0000), 
	cpu_to_be32(0x03000000), cpu_to_be32(0x180), 0x00000000, cpu_to_be32(0x180), 0x00000000, cpu_to_be32(0x00000080), 0x00000000
};

const u32 pci_prop_size_value[] = {cpu_to_be32(2)};

const u32 pci_prop_address_value[] = {cpu_to_be32(3)};

const u32 pci_prop_bus_range_values[] = {cpu_to_be32(0), cpu_to_be32(0xffffffff)};

const struct property pci_properties[] = {
	{
		.name = "#address-cells",
		.length = sizeof(u32),
		.next = &pci_properties[1],
		.value = pci_prop_address_value
	},
	{
		.name = "#size-cells",
		.length = sizeof(u32),
		.next = &pci_properties[2],
		.value = pci_prop_size_value
	},
	{
		.name = "reg",
		.length = sizeof(pci_prop_reg_values),
		.next = &pci_properties[3],
		.value = pci_prop_reg_values
	},
	{
		.name = "bus-range",
		.length = sizeof(pci_prop_bus_range_values),
		.next = &pci_properties[4],
		.value = pci_prop_bus_range_values
	},
	{
		.name = "ranges",
		.length = sizeof(pci_prop_ranges_value),
		.next = &pci_properties[5],
		.value = pci_prop_ranges_value
	},
	{
		.name = "device_type",
		.length = sizeof("pci"),
		.next = NULL,
		.value = "pci"
	}
};

struct device_node pci_device_node = {
	.name = "pci-controller",
	.full_name = "pci-controller-full",
	.properties = pci_properties,
	.parent = &platform_device_node
};

extern void init_pcibus_class(void);
extern void init_pci_bus(void);
extern u32 init_gen_pcie_driver() {
	init_pci_bus();
	init_pcibus_class();
	device_register(&platform_bus);
	struct platform_device *pcie_pdev = of_device_alloc(&pci_device_node, "pci-controller", NULL);
	pcie_pdev->dev.init_name = "pci-controller";
	device_add(&pcie_pdev->dev);
	int ret = gen_pci_driver.probe(pcie_pdev);
	printk("init pcie %d\n", ret);
	return 0;
}

MODULE_LICENSE("GPL v2");
