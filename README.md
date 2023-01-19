# Description

`ns` is a library and a program that deals with nodesets.

# Nodeset definition

A nodeset is a numbered nodename such as node2 or r1esw3-port23.
By convention numbers are put between brackets when there is more than one. You can use comma (`,`) to separate each group of numbers. A group of numbers can be either a single number (`7`) or a list of numbers (`2-9`) that may be stepped (`2-20/4`). So one can write a nodeset like:`node[1,4-7,20-30/4]`. Except the computer's memory there is no limits to the number of group of numbers nor to the number of brackets. It means that `a[1-3,5]b[12-23]c[1-20/2,80]d[100-254]e[1-16,32-64,128-196]` is a valid nodeset definition.


# Program usage

## Getting help

```bash
$ ns --help
Operations on set of nodes

Usage: ns <COMMAND>

Commands:
  fold    Fold nodeset(s) (or separate nodes) into one nodeset
  expand  Expand nodeset(s) into separate nodes
  count   Count nodeset(s)
  help    Print this message or the help of the given subcommand(s)

Options:
  -h, --help  Print help information

```

## Examples

* Expanding nodes:
```bash
$ ns expand r[1-4/2]esw[1,3,5]-port[23-24]
r1esw1-port23 r1esw1-port24 r1esw3-port23 r1esw3-port24 r1esw5-port23 r1esw5-port24 r3esw1-port23 r3esw1-port24 r3esw3-port23 r3esw3-port24 r3esw5-port23 r3esw5-port24
```
* Folding nodes:
```
$ ns fold r1esw1-port23 r1esw1-port24 r1esw3-port23 r1esw3-port24 r1esw5-port23 r1esw5-port24 r3esw1-port23 r3esw1-port24 r3esw3-port23 r3esw3-port24 r3esw5-port23 r3esw5-port24
r[1,3]esw[1,3,5]-port[23-24]
```
* Counting nodes:
```bash
$ ns count r[1-4/2]esw[1,3,5]-port[23-24]
12
```
* Intersection between nodesets:
```bash
$ ns fold 'r[1-4/2]esw[1,3,5]-port[23-24]&r[3-4]esw[2-4]-port24'
r[3]esw[3]-port[24]
```
* Difference:
```bash
$ ns fold 'r[1-4/2]esw[1,3,5]-port[23-24] ! r[3-4]esw[2-4]-port24'
r[1]esw[1,3,5]-port[23-24],r[3]esw[1,5]-port[23-24],r[3]esw[3]-port[23]
```
* Union:
```bash
$ ns fold 'r[1-4/2]esw[1,3,5]-port[23-24] + r[3-4]esw[2-4]-port24'
r[1,3]esw[1,3,5]-port[23-24],r[3]esw[2,4]-port[24],r[4]esw[2-4]-port[24]
```

