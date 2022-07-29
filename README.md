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

1. Please login to our cluster following instruction in [this page](https://github.com/hku-systems/cronus/blob/main/docs/servers.md) or setup Cronus on your own cluster using instruction in [this page](https://github.com/hku-systems/cronus/blob/main/docs/config.md).
2. Please login to server 23 (jianyu@202.45.128.182) and go to `/home/micro22ae/cronus/` to run all the experiments, where we have set up all necessary environments.
3. Each experiment will generate a figure in the `./figure` directory. You can download all generated figures to your computer by running `python3 ./tunnel.sh [private_key]` **on your computer**, which start an ssh tunnel and copy all files in `./figure` to your computer using `scp`.
4. When the script is running, you may see `END` multiple times. The script is still running; please do not suspend the script.

Please be noted that different evaluators cannot run experiments at the same time. This is because there is only one VM instance for each evaluation setup (XXX, XXX and Cronus). You can check whether other evaluators are running the experiments by XXX.

### Experiment 0: A minimum working example (CUDA matrix addition)

### Experiment 1: End-to-end performance

#### Experiment 1-1: Performance of Cronus and baseline systems (50 mins)

XXX

**Command to run:**

```shell
bash run.sh performance
```

**Output:**

- XXX

**Expected results:**

- XXX

### Create the docker images

**Prerequisite:**

1. XXX

**Configurations:**

1. XXX

2. XXX