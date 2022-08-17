ss := "cc gcc"
sample := $(shell echo ${ss} | cut -d " " -f2)
host = $(firstword $1)
port = $(word 2,$1)
all:
	@echo "$(sample)"
