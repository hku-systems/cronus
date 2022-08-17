# SPDX-License-Identifier: GPL-2.0

# Makefile for the drm device driver.  This driver provides support for the
# Direct Rendering Infrastructure (DRI) in XFree86 4.1.0 and higher.

drm-y       :=	drm_auth.c drm_cache.c \
		drm_file.c drm_gem.c drm_ioctl.c drm_irq.c \
		drm_memory.c drm_drv.c \
		drm_sysfs.c drm_hashtab.c drm_mm.c \
		drm_crtc.c drm_fourcc.c drm_modes.c drm_edid.c \
		drm_encoder_slave.c \
		drm_trace_points.c drm_prime.c \
		drm_rect.c drm_vma_manager.c drm_flip_work.c \
		drm_modeset_lock.c drm_atomic.c drm_bridge.c \
		drm_framebuffer.c drm_connector.c drm_blend.c \
		drm_encoder.c drm_mode_object.c drm_property.c \
		drm_plane.c drm_color_mgmt.c drm_print.c \
		drm_dumb_buffers.c drm_mode_config.c drm_vblank.c \
		drm_syncobj.c drm_lease.c drm_writeback.c drm_client.c \
		drm_client_modeset.c drm_atomic_uapi.c drm_hdcp.c \
		drm_managed.c drm_vblank_work.c

drm-$(CONFIG_DRM_LEGACY) += drm_legacy_misc.c drm_bufs.c drm_context.c drm_dma.c drm_scatter.c drm_lock.c
drm-$(CONFIG_DRM_LIB_RANDOM) += lib/drm_random.c
drm-$(CONFIG_DRM_VM) += drm_vm.c
drm-$(CONFIG_COMPAT) += drm_ioc32.c
drm-$(CONFIG_DRM_GEM_CMA_HELPER) += drm_gem_cma_helper.c
drm-$(CONFIG_DRM_GEM_SHMEM_HELPER) += drm_gem_shmem_helper.c
drm-$(CONFIG_DRM_PANEL) += drm_panel.c
drm-$(CONFIG_OF) += drm_of.c
drm-$(CONFIG_AGP) += drm_agpsupport.c
drm-$(CONFIG_PCI) += drm_pci.c
drm-$(CONFIG_DEBUG_FS) += drm_debugfs.c drm_debugfs_crc.c
drm-$(CONFIG_DRM_LOAD_EDID_FIRMWARE) += drm_edid_load.c

drm_vram_helper-y := drm_gem_vram_helper.c
srcs-$(CONFIG_DRM_VRAM_HELPER) += $(drm_vram_helper-y)

drm_ttm_helper-y := drm_gem_ttm_helper.c
srcs-$(CONFIG_DRM_TTM_HELPER) += $(drm_ttm_helper-y)

drm_kms_helper-y := drm_bridge_connector.c drm_crtc_helper.c drm_dp_helper.c \
		drm_dsc.c drm_probe_helper.c \
		drm_plane_helper.c drm_dp_mst_topology.c drm_atomic_helper.c \
		drm_kms_helper_common.c drm_dp_dual_mode_helper.c \
		drm_simple_kms_helper.c drm_modeset_helper.c \
		drm_scdc_helper.c drm_gem_framebuffer_helper.c \
		drm_atomic_state_helper.c drm_damage_helper.c \
		drm_format_helper.c drm_self_refresh_helper.c

drm_kms_helper-$(CONFIG_DRM_PANEL_BRIDGE) += bridge/panel.c
drm_kms_helper-$(CONFIG_DRM_FBDEV_EMULATION) += drm_fb_helper.c
drm_kms_helper-$(CONFIG_DRM_KMS_CMA_HELPER) += drm_fb_cma_helper.c
drm_kms_helper-$(CONFIG_DRM_DP_AUX_CHARDEV) += drm_dp_aux_dev.c
drm_kms_helper-$(CONFIG_DRM_DP_CEC) += drm_dp_cec.c

srcs-$(CONFIG_DRM_KMS_HELPER) += $(drm_kms_helper-y)
subdirs-$(CONFIG_DRM_DEBUG_SELFTEST) += selftests/

srcs-$(CONFIG_DRM)	+= $(drm-y)
subdirs-$(CONFIG_DRM_MIPI_DBI) += drm_mipi_dbi.c
subdirs-$(CONFIG_DRM_MIPI_DSI) += drm_mipi_dsi.c
srcs-$(CONFIG_DRM_PANEL_ORIENTATION_QUIRKS) += drm_panel_orientation_quirks.c
# subdirs-y			+= arm/
subdirs-$(CONFIG_DRM_TTM)	+= ttm/
subdirs-$(CONFIG_DRM_SCHED)	+= scheduler/
subdirs-$(CONFIG_DRM_TDFX)	+= tdfx/
subdirs-$(CONFIG_DRM_R128)	+= r128/
subdirs-$(CONFIG_DRM_RADEON)+= radeon/
subdirs-$(CONFIG_DRM_AMDGPU)+= amd/amdgpu/
subdirs-$(CONFIG_DRM_MGA)	+= mga/
subdirs-$(CONFIG_DRM_I810)	+= i810/
subdirs-$(CONFIG_DRM_I915)	+= i915/
subdirs-$(CONFIG_DRM_MGAG200) += mgag200/
subdirs-$(CONFIG_DRM_V3D)  += v3d/
subdirs-$(CONFIG_DRM_VC4)  += vc4/
subdirs-$(CONFIG_DRM_SIS)   += sis/
subdirs-$(CONFIG_DRM_SAVAGE)+= savage/
subdirs-$(CONFIG_DRM_VMWGFX)+= vmwgfx/
subdirs-$(CONFIG_DRM_VIA)	+=via/
subdirs-$(CONFIG_DRM_VGEM)	+= vgem/
subdirs-$(CONFIG_DRM_VKMS)	+= vkms/
subdirs-$(CONFIG_DRM_NOUVEAU) +=nouveau/
subdirs-$(CONFIG_DRM_EXYNOS) +=exynos/
subdirs-$(CONFIG_DRM_ROCKCHIP) +=rockchip/
subdirs-$(CONFIG_DRM_GMA500) += gma500/
subdirs-$(CONFIG_DRM_UDL) += udl/
subdirs-$(CONFIG_DRM_AST) += ast/
subdirs-$(CONFIG_DRM_ARMADA) += armada/
subdirs-$(CONFIG_DRM_ATMEL_HLCDC)	+= atmel-hlcdc/
# subdirs-y			+= rcar-du/
subdirs-$(CONFIG_DRM_SHMOBILE) +=shmobile/
# subdirs-y			+= omapdrm/
subdirs-$(CONFIG_DRM_SUN4I) += sun4i/
# subdirs-y			+= tilcdc/
subdirs-$(CONFIG_DRM_QXL) += qxl/
subdirs-$(CONFIG_DRM_BOCHS) += bochs/
subdirs-$(CONFIG_DRM_VIRTIO_GPU) += virtio/
subdirs-$(CONFIG_DRM_MSM) += msm/
subdirs-$(CONFIG_DRM_TEGRA) += tegra/
subdirs-$(CONFIG_DRM_STM) += stm/
subdirs-$(CONFIG_DRM_STI) += sti/
subdirs-$(CONFIG_DRM_IMX) += imx/
subdirs-$(CONFIG_DRM_INGENIC) += ingenic/
subdirs-$(CONFIG_DRM_MEDIATEK) += mediatek/
subdirs-$(CONFIG_DRM_MESON)	+= meson/
# subdirs-y			+= i2c/
# subdirs-y			+= panel/
# subdirs-y			+= bridge/
subdirs-$(CONFIG_DRM_FSL_DCU) += fsl-dcu/
subdirs-$(CONFIG_DRM_ETNAVIV) += etnaviv/
subdirs-$(CONFIG_DRM_ARCPGU)+= arc/
# subdirs-y			+= hisilicon/
subdirs-$(CONFIG_DRM_ZTE)	+= zte/
subdirs-$(CONFIG_DRM_MXSFB)	+= mxsfb/
# subdirs-y			+= tiny/
subdirs-$(CONFIG_DRM_PL111) += pl111/
subdirs-$(CONFIG_DRM_TVE200) += tve200/
subdirs-$(CONFIG_DRM_XEN) += xen/
subdirs-$(CONFIG_DRM_VBOXVIDEO) += vboxvideo/
subdirs-$(CONFIG_DRM_LIMA)  += lima/
subdirs-$(CONFIG_DRM_PANFROST) += panfrost/
subdirs-$(CONFIG_DRM_ASPEED_GFX) += aspeed/
subdirs-$(CONFIG_DRM_MCDE) += mcde/
subdirs-$(CONFIG_DRM_TIDSS) += tidss/
# subdirs-y			+= xlnx/
