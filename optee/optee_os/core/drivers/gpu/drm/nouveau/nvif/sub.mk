# SPDX-License-Identifier: MIT
nvif-y := object.c
nvif-y += client.c
nvif-y += device.c
nvif-y += disp.c
nvif-y += driver.c
nvif-y += fifo.c
nvif-y += mem.c
nvif-y += mmu.c
nvif-y += notify.c
nvif-y += timer.c
nvif-y += vmm.c

# Usermode classes
nvif-y += user.c
nvif-y += userc361.c

srcs-y += $(nvif-y)