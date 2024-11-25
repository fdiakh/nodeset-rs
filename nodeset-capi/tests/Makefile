CC ?= gcc
CFLAGS ?= -Wall  -I../include
LDFLAGS ?= -L../../target/debug -lnodeset
TESTS = bin/test_suite

all: $(TESTS)

bin:
	mkdir -p $@

bin/test_suite: test_suite.c ../../target/debug/libnodeset.so ../include/nodeset.h bin
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test: $(TESTS)
	LD_LIBRARY_PATH=../../target/debug bin/test_suite

clean:
	rm -f $(TESTS)

.PHONY: all clean
