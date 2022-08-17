# Artifact for paper #40 Cronus: Fault-isolated, Secure and High-performance Heterogeneous Computing for Trusted Execution Environment

## Artifact summary

Cronus is a fault-isolated, secure, and high-performance heterogeneous computing system for Trusted Execution Environment. It carries a new OS design for enabling security isolation and fault isolation, using existing TEE hardware primitives. This artifact contains the implementation of Cronus on TrustZone (QEMU) and GPU (NVIDIA)/NPU(TVM VTA) and scripts for reproducing the main results of this work.

## Artifact Check-list

- Code link: <https://github.com/hku-systems/Cronus>
- OS Version: Ubuntu 16.04 or 18.04.
- Metrics: execution time and throughput.
- Expected runtime: each trial of each data point takes about 2 xxx.

## Experiments

If otherwise specified, all Cronus's experiments run on one NVIDIA 2080 Ti GPU or a simulated VTA device.

### Prepare

1. Please login to our cluster following instruction in [this page](https://github.com/hku-systems/cronus/blob/main/docs/server.md) or setup Cronus on your own cluster using instruction in [this page](https://github.com/hku-systems/cronus/blob/main/docs/config.md).
2. Please login to the evaluation server (localhost:2233 after setting up the ssh tunnel in step 1) and go to `/home/jianyu/` to run all the experiments, where we have set up all necessary environments.

Please be noted that different evaluators cannot run experiments at the same time. This is because there is only one VM instance for each evaluation setup. You can check whether other evaluators are running the experiments by using ```ps -aux|grep qemu```.

### Troubleshooting

1. If you see `Address already in use` when running the experiment, this may be caused by others running the experiments at the same time.
2. If the program takes too long to run (e.g., more than 1min) or there are crashes, please reboot the machine. Please also make sure ```check_cpu``` returns ```4``` to ensure that there are multiple CPU cores for evaluations running on Cronus VM.
3. If the program returns a long execution time (e.g., 40s), please reboot the machine.

### Major Claims

1. Cronus supports general accelerators (GPU, NPU) and programs (Experiment 1, Experiment 2 and Experiment 4)
2. Cronus incurs a low performance overhead compared with OPTEE (TrustZone) (Experiment 1-1 and Experiment 1-2, Experiment 4-1 and Experiment 4-2)
3. Cronus is much faster than HIX-Trustzone (Experiment 1-1 and 1-3)
4. Cronus's spatial sharing of GPU improves performance during resource (GPU) contention (Experiment 3)

### Kick-off Functional (CUDA matrix addition)

**Command to run:**

1. Open one terminal (T1) and connect to the machine, create the normal OS terminal

```shell
./cronus/soc_term/soc_term 54310

```

2. Open one terminal (T2) and connect to the machine, create the secure OS terminal
```shell
./cronus/soc_term/soc_term 54311

```

3. Open one terminal (T3) and connect to the machine, and start the VM
```shell
cd cronus/build
sudo make run-only-bg
(qemu) c
```

4. Wait for the OS to boot up, after booting up, at T1, use `root` as username to login the vm 

5. Check the CPUs have been properly booted up

```shell
check_cpu
# a proper bootup returns 4, please reboot the VM if it returns 1
```

6. start the test program

```shell
cuda_test
```

5. If you see the result is ```CUDA Result is 1```, then the program works smoothly

### Experiment 1: End-to-end performance of Rodinia (30 mins)

This experiment evaluate Cronus's and Optee's performance on Rodinia.

#### Experiment 1-1: End-to-end performance of Rodinia in Cronus

**Command to run:**

Same as the Tick-off experiments, please boot up Cronus VM first, specifically as as followes

1. Open one terminal (T1) and connect to the machine, create the normal OS terminal

```shell
./cronus/soc_term/soc_term 54310

```

2. Open one terminal (T2) and connect to the machine, create the secure OS terminal
```shell
./cronus/soc_term/soc_term 54311

```

3. Open one terminal (T3) and connect to the machine, and start the VM
```shell
cd cronus/build
sudo make run-only-bg
(qemu) c
```

4. Wait for the OS to boot up, after booting up, at T1, use `root` as username to login the vm 

5. Make sure that the CPUs have been properly booted up

```shell
check_cpu
# a proper bootup returns 4, please reboot the VM if it returns 1
```

6. start the rodinia benchmark (9 programs)
```shell
# these program will usually runs less than 60s, so a long execution time suggests possible bugs in the system, please reboot the machine for solving the problem (see troubleshooting).
rodinia_bp
rodinia_bfs
rodinia_gs
rodinia_hs
rodinia_lud
rodinia_nn
rodinia_nw
rodinia_pf
rodinia_srad
```

**Output:**

- Each program will output the execution time (in seconds). Note that we measure the computation time of the program (i.e., the time for loading data from disks is not included), so the measured time is shorter than the execution of the whole program

#### Experiment 1-2: End-to-end performance of Rodinia in OPTEE (TrustZone)

**Command to run:**

Similar to the experiment 1-1, please boot up Optee VM first, specifically as as followes

1. Open one terminal (T1) and connect to the machine, create the normal OS terminal

```shell
./optee/soc_term/soc_term 54310

```

2. Open one terminal (T2) and connect to the machine, create the secure OS terminal
```shell
./optee/soc_term/soc_term 54311

```

3. Open one terminal (T3) and connect to the machine, and start the VM
```shell
cd optee/build
sudo make run-only-bg
(qemu) c
```

4. Wait for the OS to boot up, after booting up, at T1, use `root` as username to login the vm and start the rodinia benchmark (9 programs)

```shell
# these programs will usually runs less than 60s, so a long execution time suggests possible bugs in the system, please reboot the machine for solving the problem (see troubleshooting).
rodinia_bp
rodinia_bfs
rodinia_gs
rodinia_hs
rodinia_lud
rodinia_nn
rodinia_nw
rodinia_pf
rodinia_srad
```

**Output:**

- Each program will output the execution time (in seconds)

#### Experiment 1-3: End-to-end performance of Rodinia in HIX-TrustZone

Please runs experiments using the same steps as OPTEE (TrustZone), but in ```optee-hix``` (```/home/jianyu/optee-hix``` in the evaluation machine).

**Expected results in Experiment-1:**

- The execution of Rodinia in Cronus incurs moderate performance overhead compared with Optee.


### Experiment 2: End-to-end performance of DNN Training (30 mins)

#### Experiment 2-1: end-to-end performance of MNIST (LeNet) in Cronus

Step 1-5 is the same as experiment 1-1, in Step 6, execute the following commands:
```shell
dnn_mnist
```

#### Experiment 2-2: end-to-end performance of MNIST (LeNet) in OPTEE (TrustZone)

Step 1-3 is the same as experiment 1-2, in Step 4, execute the following commands:
```shell
dnn_mnist
```

**Expected results in Experiment-2:**

- Training MNIST in Cronus incurs moderate performance overhead compared with OPTEE (TrustZone).


### Experiment 3: Spatial sharing of GPU in DNN training in OPTEE (TrustZone)/Cronus

For both OPTEE (TrustZone) and Cronus, Step 1-5 is the same as experiment 1-1 (except that the the directory for running the experiment is different). Note that the machine has only four cores, so it is expected that running more than 2 programs (as Cronus requires extra threads for each mEnclave).

```shell
# (change 2 to the number of concurrent enclaves)
dnn_mnist_spatial 2
```

**Expected results in Experiment-3:**

- Spatially sharing a GPU improves utilizations of GPU, and incurs moderate overhead for each training tasks


### Experiment 4: End-to-end performance of VTA-Bench in OPTEE (TrustZone)/Cronus

#### Experiment 4-1: End-to-end performance of VTA-Bench in Cronus

Step 1-5 is the same as experiment 1-1, in Step 6, execute the following commands:
```shell
vta_bench
```

#### Experiment 4-2: End-to-end performance of VTA-Bench in OPTEE (TrustZone)

Step 1-3 is the same as experiment 1-2, in Step 4, execute the following commands:
```shell
vta_bench
```

**Expected results in Experiment-4:**

- Executing VTA-Bench in Cronus incurs moderate performance overhead compared with OPTEE (TrustZone).


