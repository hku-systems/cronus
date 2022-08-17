
ARCH ?= aarch64
CUR_DIR ?= $(shell pwd)

.PHONY : arch-$(ARCH) clean

GENERIC_FILE := $(subst generic,$(ARCH),$(wildcard generic/bits/*.h))

arch-$(ARCH): $(ARCH)/bits/alltypes.h $(ARCH)/bits/syscall.h $(GENERIC_FILE)

$(ARCH)/bits/alltypes.h: $(ARCH)/bits/alltypes.h.in ../include/alltypes.h.in ../tools/mkalltypes.sed
	sed -f ../tools/mkalltypes.sed $(ARCH)/bits/alltypes.h.in ../include/alltypes.h.in > $@

$(ARCH)/bits/syscall.h: $(ARCH)/bits/syscall.h.in
	cp $< $@
	sed -n -e s/__NR_/SYS_/p < $< >> $@

$(ARCH)/bits/%.h: generic/bits/%.h
	rm $(CUR_DIR)/$@
	ln -s $(CUR_DIR)/$< $(CUR_DIR)/$@

# obj/src/internal/version.h: $(wildcard $(srcdir)/VERSION $(srcdir)/.git)
# 	printf '#define VERSION "%s"\n' "$$(cd $(srcdir); sh tools/version.sh)" > $@

clean:
	rm -rf $(ARCH)/bits/alltypes.h $(ARCH)/bits/syscall.h

show:
	echo $(GENERIC_FILE)
