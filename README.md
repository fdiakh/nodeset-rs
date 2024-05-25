# Description

`ns` is a Rust library and command-line tool that deals with nodesets. It is
heavily inspired by [clustershell](https://cea-hpc.github.io/clustershell/) and
aims at supporting the same nodeset representation.

A nodeset is a set of names which are generally indexed over one or more integer
dimensions such as `node1` or `r1sw2-port0`. A nodeset can be represented using
a compact *folded* representation with brackets, for example
`r[1-2]sw[1-2]-port[0-1]` represents 8 names:
`r1sw1-port0,r1sw1-port1,r1sw2-port0,...,r2sw2-port1`.

`ns` allows to fold or expand nodesets, as well as to perform algebric
operations on them (union, intersection, difference, ...)

# Command line examples

* Expanding nodes:
```bash
$ ns expand r[1-4/2]esw[1,3,5]-port[23-24]
r1esw1-port23 r1esw1-port24 r1esw3-port23 r1esw3-port24 r1esw5-port23 r1esw5-port24 r3esw1-port23 r3esw1-port24 r3esw3-port23 r3esw3-port24 r3esw5-port23 r3esw5-port24
```
* Folding nodes:
```bash
$ ns fold r1esw1-port23 r1esw1-port24 r1esw3-port23 r1esw3-port24 r1esw5-port23 r1esw5-port24 r3esw1-port23 r3esw1-port24 r3esw3-port23 r3esw3-port24 r3esw5-port23 r3esw5-port24
r[1,3]esw[1,3,5]-port[23-24]
```
* Counting nodes:
```bash
$ ns count r[1-4/2]esw[1,3,5]-port[23-24]
12
```
* Ensemblist operations using operators:
```bash
$ ns fold 'r[1-4/2]esw[1,3,5]-port[23-24] & r[3-4]esw[2-4]-port24'
r[3]esw[3]-port[24]
```

# Configuration files and groups
`ns` understands and uses clustershell's configuration files in which node groups can be defined. Please refer to clustershell's documentation for a full description of the supported syntax.

# Library usage example

To compute and display the intersection of two nodesets

```rust
    use ns::NodeSet;

    let ns1: NodeSet = "node[01-15]".parse().unwrap();
    let ns2: NodeSet = "node[10-30/2]".parse().unwrap();
    let inter = ns1.intersection(&ns2);
    assert_eq!(inter.to_string(), "node[10,12,14]");
```

# Why use this crate

This project is not as mature as clustershell's nodeset and has fewer features.
However, compared to clustershell's `nodeset` tool written in Python, `ns`
starts much faster as it doesn't rely on an interpreter. It can save a lot of
time when performing multiple operations on nodesets in a shell script. It is
also faster at parsing and performing operations on nodesets. The library crate
can also be used to handle nodesets natively in Rust while sharing group
definitions with clustershell.
