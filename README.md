# Description

`ns` is a library and a program that deals with nodesets. It is heavily inspired by [clustershell](https://cea-hpc.github.io/clustershell/). The library offers rust programs the ability to manage and use nodesets for their purpose.


# Nodeset definition

A nodeset is a numbered nodename such as node2, r1esw3-port23 or node001.

By convention numbers are put between brackets when there is more than one (ie node[1-7]). You can use comma (`,`) to separate each group of numbers. A group of numbers can be either a single number (`7`) or a list of numbers (`2-9`) that may be stepped (`02-20/4`). So one can write a nodeset like:`node[1,4-7,20-30/4]`. Except the computer's memory there is no limits to the number of group of numbers nor to the number of brackets. It means that `a[1-3,5]b[12-23]c[1-20/2,80]d[100-254]e[01-16,32-64,128-196]` is a valid nodeset definition that will expand 9 657 120 node names from `a1b12c1d100e01` to `a5b23c80d254e196`.


# Library usage

To use the library you should add `ns = "0.1.0"` in the dependencies section of Cargo.toml file.

## Intersection of nodesets

Here is an example of two nodesets [intersection](`NodeSet::intersection`). One can also do:
  * union between nodesets using [extend](`NodeSet::extend`) method
  * difference using [difference](`NodeSet::difference`) method
  * symmetric_difference using [symmetric_difference](`NodeSet::symmetric_difference`) method

```rust
    use ns::NodeSet;

    let ns1: NodeSet = "node[01-15]".parse().unwrap();
    let ns2: NodeSet = "node[10-30/2]".parse().unwrap();
    let inter = ns1.intersection(&ns2);
    assert_eq!(inter.to_string(), "node[10,12,14]");
```


# Program usage

`ns` understands and uses clustershell's configuration files but one may need to replace `:` with `=` in its INI style configuration files such as `groups.conf` (this is compatible with clustershell).

## Getting help

Help is printed when invoking the program with `-h` or `--help` option. You can print help for any command invoking `help` command itself (`ns help help` returns the help of the `help` command).

## Examples

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
* Intersection between nodesets using `&` operator:
```bash
$ ns fold 'r[1-4/2]esw[1,3,5]-port[23-24] & r[3-4]esw[2-4]-port24'
r[3]esw[3]-port[24]
```
* Difference using `!` operator:
```bash
$ ns fold 'r[1-4/2]esw[1,3,5]-port[23-24] ! r[3-4]esw[2-4]-port24'
r[1]esw[1,3,5]-port[23-24],r[3]esw[1,5]-port[23-24],r[3]esw[3]-port[23]
```
* Union using `+` operator:
```bash
$ ns fold 'r[1-4/2]esw[1,3,5]-port[23-24] + r[3-4]esw[2-4]-port24'
r[1,3]esw[1,3,5]-port[23-24],r[3]esw[2,4]-port[24],r[4]esw[2-4]-port[24]
```
