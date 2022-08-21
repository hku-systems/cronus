
CUR_DIR ?= $(shell pwd)

.PHONY : clean

GENERIC_FILE := $(subst ../cronus,$(CUR_DIR),$(wildcard ../cronus/*))
GENERIC_DLELETE_FILE := $(subst $(CUR_DIR),$(CUR_DIR)/delete/,$(wildcard $(CUR_DIR)/*))

all: $(GENERIC_FILE)

$(CUR_DIR)/%: ../cronus/%
	@test ! -d $(CUR_DIR)/$< || ln -s $(CUR_DIR)/$< $@

$(CUR_DIR)/delete/%: $(CUR_DIR)/%
	@test ! -L $< || rm $<

clean: $(GENERIC_DLELETE_FILE)

show:
	@echo $(GENERIC_FILE)
