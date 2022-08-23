
## Developing Apps in Cronus

For an application, create a new directory in ```optee_examples``` (following the structure in rodinia_bp).

The ```host``` directory stores the untrusted code of an application; the ```ta``` stores the code for a monolithic enclave; ```cpumenclave``` stores the code of CPU mEnclave; ```cudamenclave``` stores the code of CUDA mEnclave; ```vtamencalve``` stores the code of NPU mEnclave.

For an application, the developer needs to port the trusted code within enclave and stores it in ```ta``` (for TrustZone, i.e., OPTEE). To port the enclave code to Cronus, the developer replicate the code in ```ta``` into two directories ```cpumenclave``` and ```cudamenclave```, where they differ only in ```sub.mk```. cpumenclave's sub.mk includes ```../../cpuenclave.mk```; while cudamenclave's sub.mk includes ```../../cudaenclave.mk```. 

The above steps are similar in porting NPU menclaves. Please see ```vta_bench``` for more details.