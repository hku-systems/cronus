
CUR_DIR ?= $(shell pwd)/..

all: aarch64-linux-gnu-nvcc

install: aarch64-linux-gnu-nvcc
	@echo "Copying $<"
	@cp $< /usr/local/cuda-11.4/bin/

aarch64-linux-gnu-nvcc: aarch64-linux-gnu-nvcc.template
	@echo "Generaing $@"
	@sed 's#GCC_DIR#${CUR_DIR}#g' aarch64-linux-gnu-nvcc.template > $@
	@chmod +x $@

clean:
	@rm aarch64-linux-gnu-nvcc
