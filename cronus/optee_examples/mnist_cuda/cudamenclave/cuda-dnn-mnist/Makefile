SOURCE_DIR = src
DATA_DIR = data
BUILD_DIR = build
LOGS_DIR = logs
EXEC_FILE = CUDA-DNN-MNIST

CPU_SOURCE_FILES := $(shell find $(SOURCEDIR) -name '*.cpp')
CPU_OBJ_FILES := $(CPU_SOURCE_FILES:%=${BUILD_DIR}/%.o)
GPU_SOURCE_FILES := $(shell find $(SOURCEDIR) -name '*.cu')
GPU_OBJ_FILES := $(GPU_SOURCE_FILES:%=${BUILD_DIR}/%.o)

dataset:
	mkdir -p ${DATA_DIR}
	curl -o ${DATA_DIR}/train-images.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz 
	curl -o ${DATA_DIR}/train-labels.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz 
	curl -o ${DATA_DIR}/test-images.gz http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
	curl -o ${DATA_DIR}/test-labels.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
	gunzip ${DATA_DIR}/train-images.gz
	gunzip ${DATA_DIR}/train-labels.gz
	gunzip ${DATA_DIR}/test-images.gz
	gunzip ${DATA_DIR}/test-labels.gz

${BUILD_DIR}/%.cu.o: %.cu
	nvcc -I. -arch sm_75 -c $< -lineinfo -o $@

${BUILD_DIR}/%.cpp.o: %.cpp
	g++ -c $< -o $@

build: $(GPU_OBJ_FILES) $(CPU_OBJ_FILES)
	mkdir -p ${BUILD_DIR}
	g++ -o ${BUILD_DIR}/runapp $^ -lucuda -L/usr/local/gdev/lib64/ -lgdev

run:
	mkdir -p ${LOGS_DIR}
	sudo ./${BUILD_DIR}/runapp


run_experiments:
	mkdir -p ${LOGS_DIR}
	python3 run_experiments.py

clean:
	rm -rf ${BUILD_DIR}

FORCE:
