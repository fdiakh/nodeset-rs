CC ?= gcc
CFLAGS ?= -Wall -Werror -I../include
LDFLAGS ?= -L../../target/debug -lnodeset
EXAMPLES = bin/ns_list bin/ns_intersect_iter

all: $(EXAMPLES)

bin:
	mkdir -p $@

bin/ns_list: ns_list.c ../../target/debug/libnodeset.so ../include/nodeset.h bin
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

bin/ns_intersect_iter: ns_intersect_iter.c ../../target/debug/libnodeset.so ../include/nodeset.h bin
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test: $(EXAMPLES)
	LD_LIBRARY_PATH=../../target/debug bin/ns_list 'n[01-10]!n[02-09]' | xargs echo | grep -qx 'n01 n10'
	LD_LIBRARY_PATH=../../target/debug bin/ns_intersect_iter 'n[01-08]' 'n[06-10]' | xargs echo | grep -qx 'n06 n07 n08'

clean:
	rm -f $(EXAMPLES)

.PHONY: all clean
