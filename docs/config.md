
# How to setup Cronus on a machine

## Hardware requirements
- CPU: ARM/Intel CPU
- GPU: NVIDIA 2080Ti

## Software requirements
- OS: Ubuntu 18.04
- CUDA (NVCC): 11.4 
- QEMU: 5.2.50 (integrated in the code)

## Setup steps

1. Install CUDA (NVCC) to /usr/local/cuda-11.4

2. Configure the NVIDIA GPU Driver for passthrough
    - Use ```lspci -vvv``` to check the PCI ID of the NVIDIA GPU
    - create ```/etc/modprobe.d/vfio.conf``` using the following content, using the id from the previous step
        ```
            softdep vfio-pci pre: vfio
            softdep nouveau pre: vfio-pci
            softdep nvidiafb pre: vfio-pci
            softdep snd_hda_intel pre: vfio-pci
            softdep nvidia pre: vfio-pci
            softdep i2c_nvidia_gpu pre: vfio-pci
            softdep xhci_hcd pre: vfio-pci
            options vfio-pci ids=10de:1e07,10de:10f7,10de:1ad6,10de:1ad7
        ```
    - reboot the system and check if the configuration is ready
    - in some case, you can also use the following command to override the driver of a GPU device
        ```
        echo 0000:3b:00.2 > /sys/bus/pci/devices/0000\:3b\:00.2/driver/unbind
        echo "vfio-pci" > /sys/bus/pci/devices/0000\:3b\:00.2/driver_override
        ```
    - a successfully configuration is as follows
        ```
        3b:00.0 VGA compatible controller: NVIDIA Corporation GV102 (rev a1) (prog-if 00 [VGA controller])
            Subsystem: Gigabyte Technology Co., Ltd Device 37c4
            Control: I/O+ Mem+ BusMaster- SpecCycle- MemWINV- VGASnoop- ParErr+ Stepping- SERR+ FastB2B- DisINTx-
            Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
            Interrupt: pin A routed to IRQ 40
            NUMA node: 0
            Region 0: Memory at b7000000 (32-bit, non-prefetchable) [size=16M]
            Region 1: Memory at 38bfe0000000 (64-bit, prefetchable) [size=256M]
            Region 3: Memory at 38bff0000000 (64-bit, prefetchable) [size=32M]
            Region 5: I/O ports at 7000 [size=128]
            Expansion ROM at b8000000 [disabled] [size=512K]
            Capabilities: <access denied>
            Kernel driver in use: vfio-pci
            Kernel modules: nvidiafb, nouveau, nvidia_drm, nvidia

        3b:00.1 Audio device: NVIDIA Corporation Device 10f7 (rev a1)
            Subsystem: Gigabyte Technology Co., Ltd Device 37c4
            Control: I/O- Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr+ Stepping- SERR+ FastB2B- DisINTx-
            Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
            Latency: 0, Cache Line Size: 32 bytes
            Interrupt: pin B routed to IRQ 10
            NUMA node: 0
            Region 0: Memory at b8080000 (32-bit, non-prefetchable) [size=16K]
            Capabilities: <access denied>
            Kernel driver in use: vfio-pci
            Kernel modules: snd_hda_intel

        3b:00.2 USB controller: NVIDIA Corporation Device 1ad6 (rev a1) (prog-if 30 [XHCI])
            Subsystem: Gigabyte Technology Co., Ltd Device 37c4
            Control: I/O- Mem+ BusMaster- SpecCycle- MemWINV- VGASnoop- ParErr+ Stepping- SERR+ FastB2B- DisINTx-
            Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
            Interrupt: pin C routed to IRQ 40
            NUMA node: 0
            Region 0: Memory at 38bff2000000 (64-bit, prefetchable) [size=256K]
            Region 3: Memory at 38bff2040000 (64-bit, prefetchable) [size=64K]
            Capabilities: <access denied>
            Kernel driver in use: vfio-pci

        3b:00.3 Serial bus controller [0c80]: NVIDIA Corporation Device 1ad7 (rev a1)
            Subsystem: Gigabyte Technology Co., Ltd Device 37c4
            Control: I/O- Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr+ Stepping- SERR+ FastB2B- DisINTx-
            Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
            Latency: 0, Cache Line Size: 32 bytes
            Interrupt: pin D routed to IRQ 11
            NUMA node: 0
            Region 0: Memory at b8084000 (32-bit, non-prefetchable) [size=4K]
            Capabilities: <access denied>
            Kernel driver in use: vfio-pci
        ```
3. clone the source code and run the following command to build the software
    ```
    cd /path/to/cronus/build
    make -j40
    ```

4. configure the passthrough configuration in the Makefile
   ```
   # change the "3b:00.0" in following line in the run-only-bg target, using the information in Step 2
   device vfio-pci,host=3b:00.0,bus=pcie.0
   ```