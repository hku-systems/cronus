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
3. Each experiment will generate a figure in the `./figure` directory. You can download all generated figures to your computer by running `python3 ./tunnel.sh [private_key]` **on your computer**, which start an ssh tunnel and copy all files in `./figure` to your computer using `scp`.
4. When the script is running, you may see `END` multiple times. The script is still running; please do not suspend the script.

Please be noted that different evaluators cannot run experiments at the same time. This is because there is only one VM instance for each evaluation setup. You can check whether other evaluators are running the experiments by using ```ps -aux|grep qemu```.

### Kick-off Functional (CUDA matrix addition)

**Troubleshooting**

1. If you see `Address already in use` when running the experiment, this may be caused by others running the experiments at the same time.

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

4. Wait for the OS to boot up, after booting up, at T1, use `root` as username to login the vm and start the test program

```shell
cronus_cuda
```

5. If you see the result is ```CUDA Result is 1```, then the program works smoothly

### Experiment 1: End-to-end performance of Rodinia (30 mins)

This experiment evaluate Cronus's and Optee's performance on Rodinia.

#### Experiment 1-1: End-to-end performance of Rodinia in Cronus

**Command to run:**

Same as the Tick-off experiments, please boot up Cronus first, specificallyas as followes

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

4. Wait for the OS to boot up, after booting up, at T1, use `root` as username to login the vm and start the rodinia benchmark (9 programs)

```shell
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

#### Experiment 1-1: End-to-end performance of Rodinia in Cronus

**Command to run:**

Same as the Tick-off experiments, please boot up Cronus first, specificallyas as followes

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


**Expected results:**

- The execution of Rodinia in Cronus incurs moderate performance overhead compared with Optee.


### Experiment 2: End-to-end performance of DNN Training (30 mins)
