# Define variables
CARGO_BIN ?= cargo
BINDGEN_BIN ?= cbindgen
BUILD_DIR ?= target/debug

# Default target
all: nodeset examples

nodeset:
	$(CARGO_BIN) build

test: tests examples
	cargo +nightly miri test
	$(MAKE) -C tests test
	$(MAKE) -C examples test

tests: nodeset
	$(MAKE) -C tests

bindgen:
	 $(BINDGEN_BIN) -c cbindgen.toml --crate nodeset-capi --output include/nodeset.h

examples: nodeset
	$(MAKE) -C examples

# Clean build artifacts
clean:
	$(CARGO_BIN) clean
	$(MAKE) -C examples clean
	$(MAKE) -C tests clean

.PHONY: all build tests examples clean nodeset
