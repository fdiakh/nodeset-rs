#![cfg_attr(feature = "unstable", feature(test))]

use regex::Regex;
use std::collections::HashMap;
use fnv::{FnvHashSet, FnvBuildHasher};
use std::collections::hash_set::{
    Difference as HSDifference,
    Union as HSUnion,
    Intersection as HSIntersection,
    SymmetricDifference as HSSymmetricDifference};
//use std::io;
use rand::prelude::*;
use itertools::Itertools;

//todo: implement partialEq
#[derive(Debug, PartialEq, Clone)]
pub struct IdRangeList {
    indexes: Vec<u32>,
    sorted: bool
}

#[derive(Debug, PartialEq, Clone)]
pub struct IdRangeSet {
    indexes: FnvHashSet<u32>
}

struct VecDifference<'a, T> {
    a: std::slice::Iter<'a, T>,
    b: std::iter::Peekable<std::slice::Iter<'a, T>>
}

struct VecIntersection<'a, T> {
    a: &'a[T],
    b: &'a[T]
}

struct VecUnion<'a, T> {
    a: &'a[T],
    b: &'a[T]
}

struct VecSymDifference<'a, T> {
    a: &'a[T],
    b: &'a[T]
}

impl IdRangeSet {
}

impl IdRangeList {
    fn sort(self: &mut Self) {
        self.indexes.sort();
        self.sorted = true;
    }
}

impl<'a, T> Iterator for VecDifference<'a, T>
    where T: Ord + std::fmt::Debug
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let mut next = match self.a.next() {
            Some(v) => v,
            None => return None
        };

        let mut min: &T;
        loop {
            min = match self.b.peek() {
                None    => return Some(next),
                Some(v) if *v == next => v,
                Some(v) if *v > next => return Some(next),
                _ => {self.b.next(); continue}

            };

            while next == min {
                next = match self.a.next() {
                    Some(v) => v,
                    None => return None
                };
            }
       }
    }
}

impl<'a, T> Iterator for VecIntersection<'a, T>
    where T: Ord + std::fmt::Debug
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        while !self.a.is_empty() && !self.b.is_empty() {
            let first_a = self.a.first();
            let first_b = self.b.first();

            if first_a == first_b {
                self.a = &self.a[1..];
                self.b = &self.b[1..];
                return first_a;
            } else if first_a < first_b{
                    self.a = &self.a[exponential_search_idx(self.a, &first_b.unwrap())..];
            } else {
                    self.b = &self.b[exponential_search_idx(self.b, &first_a.unwrap())..];
            }
        }

        return None
    }
}

impl<'a, T> Iterator for VecUnion<'a, T>
    where T: Ord + std::fmt::Debug
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let first_a = self.a.first();
        let first_b = self.b.first();

        if first_a != None && (first_a <= first_b || first_b == None) {
                self.a = &self.a[1..];
                return first_a;
        }

        if first_b != None {
            self.b = &self.b[1..];
            return first_b;
        }

        return None;
    }
}

impl<'a, T> Iterator for VecSymDifference<'a, T>
    where T: Ord + std::fmt::Debug
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {

        let mut first_a = self.a.first();
        let mut first_b = self.b.first();

        while first_a == first_b {
            if first_a == None {
                return None
            }
            self.a = &self.a[1..];
            self.b = &self.b[1..];
            first_a = self.a.first();
            first_b = self.b.first();
        }

        if first_a != None && (first_a < first_b || first_b == None) {
                self.a = &self.a[1..];
                return first_a;
        }
        self.b = &self.b[1..];
        return first_b;
    }
}

trait IdRange<'a> {
    type DifferenceIter: Iterator<Item = &'a u32>;
    type SymmetricDifferenceIter: Iterator<Item = &'a u32>;
    type IntersectionIter: Iterator<Item = &'a u32>;
    type UnionIter: Iterator<Item = &'a u32>;

    fn new(indexes: Vec<u32>) -> Self;
    fn difference(self: &'a Self, other: &'a Self) -> Self::DifferenceIter;
    fn symmetric_difference(self: &'a Self, other: &'a Self) -> Self::SymmetricDifferenceIter;
    fn intersection(self: &'a Self, other: &'a Self) -> Self::IntersectionIter;
    fn union(self: &'a Self, other: &'a Self) -> Self::UnionIter;
    fn contains(self: &Self, id: u32) -> bool;
    fn is_empty(self: &Self) -> bool;
}

impl<'a> IdRange<'a> for IdRangeSet {
    type DifferenceIter = HSDifference<'a, u32, FnvBuildHasher>;
    type SymmetricDifferenceIter = HSSymmetricDifference<'a, u32, FnvBuildHasher>;
    type IntersectionIter = HSIntersection<'a, u32, FnvBuildHasher>;
    type UnionIter = HSUnion<'a, u32, FnvBuildHasher>;

    fn new(indexes: Vec<u32>) -> IdRangeSet {
        let mut bt: FnvHashSet<u32>  = FnvHashSet::with_hasher(Default::default());
        bt.extend(&indexes);
        IdRangeSet {
            indexes: bt
        }
    }
    fn difference(self: &'a Self, other: &'a Self) -> Self::DifferenceIter {
        return self.indexes.difference(&other.indexes);
    }
    fn symmetric_difference(self: &'a Self, other: &'a Self) -> Self::SymmetricDifferenceIter {
        return self.indexes.symmetric_difference(&other.indexes);
    }
    fn intersection(self: &'a Self, other: &'a Self) -> Self::IntersectionIter {
        return self.indexes.intersection(&other.indexes);
    }
    fn union(self: &'a Self, other: &'a Self) -> Self::UnionIter {
        return self.indexes.union(&other.indexes);
    }
    fn contains(self: &Self, id: u32) -> bool {
        return self.indexes.contains(&id);
    }
    fn is_empty(self: &Self) -> bool {
        return self.indexes.is_empty()
    }
}

fn exponential_search<T>(v: &[T], x: &T) -> Result<usize, usize>
    where T: Ord
{
    let mut i: usize = 1;
    while i <= v.len() {
        if v[i-1] == *x {
            return Ok(i-1);
        }
        if v[i-1] > *x {
            break
        }
        i *= 2;
    }

    match v[i/2..std::cmp::min(i-1, v.len())].binary_search(x) {
        Ok(result) => Ok(result + i/2),
        Err(result) => Err(result + i/2)
    }
}

fn exponential_search_idx<T>(v: &[T], x: &T) -> usize
    where T: Ord
{
    match exponential_search(v, x) {
        Ok(r) => r,
        Err(r) => r
    }
}

impl<'a> IdRange<'a> for IdRangeList {
    type DifferenceIter = VecDifference<'a, u32>;
    type IntersectionIter = VecIntersection<'a, u32>;
    type UnionIter = VecUnion<'a, u32>;
    type SymmetricDifferenceIter = VecSymDifference<'a, u32>;

    fn new(indexes: Vec<u32>) -> IdRangeList {
        IdRangeList {
            indexes,
            sorted: false
        }
    }

    fn contains(self: &Self, id: u32) -> bool {
        exponential_search(&self.indexes, &id).is_ok()
    }

    fn intersection(self: &'a Self, other: &'a Self) -> Self::IntersectionIter {
        Self::IntersectionIter {
            a: &self.indexes,
            b: &other.indexes
        }
    }

    fn union(self: &'a Self, other: &'a Self) -> Self::UnionIter {
        Self::UnionIter {
            a: &self.indexes,
            b: &other.indexes
        }
    }

    fn symmetric_difference(self: &'a Self, other: &'a Self) -> Self::SymmetricDifferenceIter {
        Self::SymmetricDifferenceIter {
            a: &self.indexes,
            b: &other.indexes
        }
    }

    fn difference(self: &'a Self, other: &'a Self) -> Self::DifferenceIter {
        assert!(self.sorted);
        assert!(other.sorted);
        Self::DifferenceIter {
            a: self.indexes.iter(),
            b: other.indexes.iter().peekable()
        }
    }

    fn is_empty(self: &Self) -> bool {
        return self.indexes.is_empty();
    }
}


#[derive(Debug, Clone)]
struct IdRangeProduct<T>
{
    ranges: Vec<T>
}


impl<'a, T> IdRangeProduct<T>
    where T: IdRange<'a>
{
    fn intersection(self: &'a Self, other: &'a Self) -> Option<IdRangeProduct<T>> {
        let mut ranges = Vec::<T>::new();

        for (sidr, oidr) in self.ranges.iter()
                                    .zip(other.ranges.iter()) {
            let rng = T::new(sidr.intersection(&oidr).cloned().collect::<Vec<u32>>());
            if rng.is_empty() {
                return None
            }
            ranges.push(rng);
        }

        Some(IdRangeProduct{ranges})
    }

    fn iter(self: &Self) -> std::slice::Iter<T>{
        self.ranges.iter()
    }
}

#[derive(Debug)]
struct IdSet<T> {
    products: Vec<IdRangeProduct<T>>
}

impl<T> std::fmt::Display for IdSet<T>
    where T: std::fmt::Display
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // Customize so only `x` and `y` are denoted.
        write!(f, "{}", self.products.iter().join(","))
    }
}

impl<T> std::fmt::Display for IdRangeProduct<T>
    where T: std::fmt::Display
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // Customize so only `x` and `y` are denoted.
        write!(f, "{}", self.ranges.iter().join(""))
    }
}

impl std::fmt::Display for IdRangeList
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut rngs = self.indexes.iter()
                        .zip(self.indexes.iter().chain(self.indexes.iter().last()).skip(1))
                        .batching(|it| {
                            if let Some((&first, &next)) = it.next() {
                                if next != first + 1 {
                                    return Some(first.to_string())
                                }
                                for (&cur, &next) in it {
                                    if next != cur + 1 {
                                        return Some(format!("{}-{}", first, cur))
                                    }
                                }
                                // Should never be reached
                                None
                            } else {
                                None
                            }
                    });
        write!(f, "[{}]", rngs.join(","))
    }
}

impl std::fmt::Display for IdRangeSet
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
         write!(f, "[{}]", self.indexes.iter().join(","))
    }
}

impl <'a, T> IdSet<T>
    where T: IdRange<'a> + PartialEq + Clone
{
// TODO: return Result instead of panics
// TODO: Enforce dimname for first match for nodeset
    fn push(self: &mut Self, nodelist: &str) {
        //let re = Regex::new(r"\s*(?P<dimname>@?[a-zA-Z_][a-zA-Z_\.\-]*)(?:(?P<index>[0-9]+)|(?:\[(?P<sindex>[0-9]+)(?:-(?P<eindex>[0-9]+))?(?:/(?P<step>[0-9]+))?\]))?(?P<sep>[,])?").unwrap();
        let dim_re = Regex::new(r"\s*(?P<dimname>@?[a-zA-Z_][a-zA-Z_\.\-]*)?(?:(?P<index>[0-9]+)|(?:\[(?P<range>[0-9,-/]+)\]))?(?P<sep>[,])?").unwrap();
        let range_re = Regex::new(r"(?P<sindex>[0-9]+)(?:-(?P<eindex>[0-9]+))?(?:/(?P<step>[0-9]+))?(?P<sep>[,])?").unwrap();
        let mut prev_index = 0;
        //let mut dims = 0;
        let mut ranges = Vec::<T>::new();
        for caps in dim_re.captures_iter(&nodelist) {
            let new_index = caps.get(0).unwrap().start();
            if new_index != prev_index {
                println!(
                    "Failed to parse at char {}: {}",
                    prev_index,
                    &nodelist[prev_index..new_index]
                )
            }

            if let Some(m) = caps.name("index") {
                let index: u32 = m.as_str().parse().unwrap();
                ranges.push(T::new(vec![index]));
            } else if let Some(m) = caps.name("range") {
                // println!("bracket matched");
                let mut range = Vec::<u32>::new();
                let mut no_sep = false;
                for rng_caps in range_re.captures_iter(m.as_str()) {
                    if no_sep {
                        panic!("Failed to parse at char {}: {}", prev_index, &nodelist[prev_index..new_index]);
                    }

                    if let Some(m) = rng_caps.name("eindex") {
                        // println!("range");
                        let sindex: u32 = rng_caps["sindex"].parse().unwrap();
                        let eindex: u32 = m.as_str().parse().unwrap();
                        let mut step = 1;
                        if let Some(s) = rng_caps.name("step") {
                            // println!("step");
                            step = s.as_str().parse().unwrap();
                        }
                        range.extend((sindex..eindex+1).step_by(step))
                    } else if let Some(m) = rng_caps.name("sindex") {
                        // println!("single");
                        range.push(m.as_str().parse().unwrap())
                    } else {
                        panic!("Failed to parse at char {}: {}", prev_index, &nodelist[prev_index..new_index]);
                    }

                    if rng_caps.name("sep").is_none() {
                        no_sep = true;
                    }
                } // TODO: separators, errors
                ranges.push(T::new(range));
            }

            if caps.name("sep").is_some() {
                /* if dims > 0 && dims != ranges.len() {
                    // TODO: mismatched dims, may not be an issue
                    panic!("Failed to parse at char {}: {}", prev_index, &nodelist[prev_index..new_index]);
                } */
                self.products.push(IdRangeProduct{ranges});
                ranges = Vec::<T>::new();
            }
            prev_index = caps.get(0).unwrap().end();
        }

        if !ranges.is_empty() {
            self.products.push(IdRangeProduct{ranges});
        }
    }
    fn fold(self: &'a Self) {
        let mut start = 1;
        let mut split_products: Vec<IdRangeProduct<T>> = vec![];

        if self.products.is_empty() {
            return
        } else {
            split_products.push(self.products[0].clone());
        }

        while start < self.products.len(){
            let p1 = &self.products[start];
           /* let mut new_products: Vec<IdRangeProduct<T>> = vec![]; */
            for p2 in &split_products {
                /* let mut new_p = IdRangeProduct::<T>{ranges: vec![]}; */
                for (axis, (r1, r2)) in p1.iter()
                                        .zip(p2.iter())
                                        .enumerate() {
                    if r2.intersection(r1).next().is_none() {
                        break;
                    }
/*                     if r1 == r2 {
                        new_p.ranges.push((*r1).clone());
                    } else {
                        /* new_p.ranges.push(T::new(r1.intersection(r2).cloned().collect())); */
                        if r1.difference(r2).next().is_some() {
                            let diff = vec![T::new(r1.difference(r2).cloned().collect())];
                            let new_iter = p1.ranges[0..axis].iter()
                                                             .chain(&diff)
                                                             .chain(&(p1.ranges[axis+1..]));
                            new_products.push(IdRangeProduct{ranges: new_iter.cloned().collect()});
                        }
                        if r2.difference(r1).next().is_some() {
                            let diff = vec![T::new(r2.difference(r1).cloned().collect())];
                            let new_iter = p1.ranges[0..axis].iter()
                                                             .chain(&diff)
                                                             .chain(&(p1.ranges[axis+1..]));
                            new_products.push(IdRangeProduct{ranges: new_iter.cloned().collect()});
                        }
                    } */

                }
            }
        }
    }
    fn intersection(self: &'a Self, other: &'a Self) -> Self {
        let mut products = Vec::<IdRangeProduct<T>>::new();
        for (sidpr, oidpr) in self.products.iter()
            .cartesian_product(other.products.iter()) {
                if let Some(idpr) = sidpr.intersection(oidpr) {
                    products.push(idpr)
                }
        }
        IdSet{products}
    }

    fn new() -> Self {
        IdSet{products: Vec::new()}
    }
}

pub struct NodeSet<T> {
    nodenames: HashMap<String, IdSet<T>>,
}
/*
impl NodeList {
    pub fn new() -> NodeList {
        NodeList {
            nodenames: HashMap::new(),
            noderanges: Vec::new(),
        }
    }
    fn register_node(self: &mut Self, nodename: &str) -> u16 {
        match self.nodenames.get(&vec!(String::from(nodename))) {
            Some(nodeid) => *nodeid,
            None => {
                let len: u16 = self.nodenames.len() as u16;
                self.nodenames.insert(vec!(String::from(nodename)), len);
                len
            }
        }
    }

    fn push_one(self: &mut Self, nodeid: u16, coord: &[u32]) {
        match self.noderanges.last_mut(){
            Some(ref mut nr) if nr.nodeid() == nodeid => {
                nr.push(coord);
            },
            _ => match coord.len() {
                1 => self.noderanges.push(Box::new(NodeRange::new(Coord1::from_slice(coord), nodeid))),
                _ => panic!()
            }
        }

    }


// TODO: parser [x-y,z,t,a-b] dans une deuxieme boucle
// Boucler sur: (Spaces)(Dimname)(DimCount)?Sep?
    pub fn push(self: &mut Self, nodelist: &str) {
        let re = Regex::new(r"\s*(?P<dimname>@?[a-zA-Z_][a-zA-Z_\.\-]*)(?:(?P<index>[0-9]+)|(?:\[(?P<sindex>[0-9]+)(?:-(?P<eindex>[0-9]+))?(?:/(?P<step>[0-9]+))?\]))?(?P<sep>[,])?").unwrap();
        let mut prev_index = 0;

        for caps in re.captures_iter(&nodelist) {
            let new_index = caps.get(0).unwrap().start();
            if new_index != prev_index {
                println!(
                    "Failed to parse at char {}: {}",
                    prev_index,
                    &nodelist[prev_index..new_index]
                )
            }

            let name = caps.name("dimname").unwrap().as_str();
            let nodeid = self.register_node(name);

            if let Some(m) = caps.name("index") {
                let index = m.as_str().parse().unwrap();
                self.push_one(nodeid, &[index]);
            } else if let Some(m) = caps.name("eindex") {
                let sindex: u32 = caps["sindex"].parse().unwrap();
                let eindex: u32 = m.as_str().parse().unwrap();
                let mut step = 1;
                if let Some(s) = caps.name("step") {
                    step = s.as_str().parse().unwrap();
                }
                for index in (sindex..eindex + 1).step_by(step) {
                    self.push_one(nodeid, &[index]);
                }
            }

            prev_index = caps.get(0).unwrap().end();
        }
    }

    pub fn fold(self: &mut Self) -> String {
        let mut folds = Vec::new();
        for nr in &mut self.noderanges {
            let components = self.nodenames.iter().filter(|x| *(x.1) == nr.nodeid()).next().unwrap().0;
            folds.push(nr.fold(components))
        }

        folds.join(",")
    }
}

type NodeName = Vec<String>;

trait Coord {
    fn from_slice(c: &[u32]) -> Self;
    fn null() -> Self;
    fn adjacent(self: &Self, c: &Self) -> bool;
    fn print_range(beg: &Self, end: &Self) -> String;
}

#[derive(Eq, Ord, PartialEq, PartialOrd)]
struct Coord1(u32);


impl Coord for Coord1 {
    fn from_slice(c: &[u32]) -> Self {
        Self(c[0])
    }

    fn null() -> Self {
        Self(0)
    }

    fn adjacent(self: &Self, c: &Self) -> bool {
        return self.0 + 1 == c.0
    }

    fn print_range(beg: &Self, end: &Self) -> String {
        match beg {
            e if e.0 == end.0 => format!("{}", end.0),
            _   => format!("{}-{}", beg.0, end.0)
        }
    }
}

struct NodeRange<T> {
    nodeid: u16,
    coords: Vec<T>,
}

trait NR {
    fn push(self: &mut Self, c: &[u32]);
    fn len(self: &Self) -> usize;
    fn fold(self: &mut Self, components: &NodeName) -> String;
    fn nodeid(self: &Self) -> u16;
}

impl<T: Coord + Ord> NR for NodeRange<T> {

    fn push(self: &mut Self, c: &[u32]) {
        self.coords.push(T::from_slice(c))
    }

   fn len(self: &Self) -> usize {
        self.coords.len()
    }

    fn nodeid(self: &Self) -> u16 {
        self.nodeid
    }

    fn fold(self: &mut Self, components: &NodeName) -> String {
        let mut first = true;
        let mut foldset = Vec::new();
        let mut cur_range = (&T::null(), &T::null());

        self.coords.sort();

        for c in &self.coords {
            if first {
                cur_range = (c, c);
                first = false;
            } else if cur_range.1 == c {
                continue;
            } else if cur_range.1.adjacent(&c) {
                cur_range.1 = c;
            } else {
                foldset.push(cur_range);
                cur_range = (c, c)
            }
        }

        if !first {
            foldset.push(cur_range)
        }
        let mut r = Vec::new();
        for (beg, end) in foldset {
            r.push(T::print_range(beg,end))
        }
        return format!("{}[{}]", components[0], r.join(","));
    }
}

impl<T: Coord> NodeRange<T> {
    fn new(c: T, nodeid: u16) -> NodeRange<T> {
        NodeRange {
            nodeid: nodeid,
            coords: vec![c]
        }
    }
}
 */
fn main() {
/*     let mut text = String::new();
    io::stdin()
        .read_line(&mut text)
        .expect("Failed to read line");

    let mut n = NodeList::new();
    n.push(&text);
    println!("{}", n.fold()); */

/*     let mut line = String::new();
    let mut v  : Vec<u32> = Vec::new();
    let mut v2  : Vec<u32> = Vec::new();
    let mut rng = thread_rng();
    io::stdin().read_line(&mut line).expect("Failed to read count");
    let count: u32 = line.trim().parse().expect("Failed to parse count");


    for x in 0..count {
        v.push(x);
    }
    v.shuffle(&mut rng);

    for x in 0..count {
        v2.push(x+1);
    }
    v2.shuffle(&mut rng);

    let r = v.clone();
    let r2 = v2.clone();
    let mut rl = IdRangeList::new(r);
    let mut rl2 = IdRangeList::new(r2);
    rl.sort();
    rl2.sort();

    println!("Timing in a Vec for {} elems", count);
    let start = Instant::now();


    let mut count = 0;
    for _ in 1..10000000 {
        count  += rl.intersection(&rl2).count();
    }
    println!("Duration is {:?}", start.elapsed()/10000000);

    println!("Timing in a Set for {} elems", count);
    let start = Instant::now();
    let bs = IdRangeSet::new(v);
    let bs2 = IdRangeSet::new(v2);
    println!("Difference count {:?}", bs.difference(&bs).count());
    println!("Difference count {:?}", bs.difference(&bs2).count());
    println!("Intersection count {:?}", bs.intersection(&bs).count());
    println!("Intersection count {:?}", bs.intersection(&bs2).count());
    println!("Duration is {:?}", start.elapsed());
 */


}
#[cfg(all(feature = "unstable", test))]
mod benchs {
    extern crate test;
    use test::{Bencher, black_box};
    use super::*;

    fn prepare_vector_ranges(count: u32, ranges: u32) -> Vec<u32> {
        let mut res: Vec::<u32> = Vec::new();
        for i in (0..ranges).rev() {
            res.append(&mut (count*i..count*(i+1)).collect());
        }
        return res;
    }

    fn prepare_vectors(count1: u32, count2: u32) -> (Vec<u32>, Vec<u32>) {
        let mut v1: Vec<u32> = (0..count1).collect();
        let mut v2: Vec<u32> = (1..count2+1).collect();
        let mut rng = thread_rng();

        v1.shuffle(&mut rng);
        v2.shuffle(&mut rng);
        (v1, v2)
    }

    fn prepare_rangelists(count1: u32, count2: u32) -> (IdRangeList, IdRangeList) {
        let (v1, v2) = prepare_vectors(count1, count2);
        let mut rl1 = IdRangeList::new(v1.clone());
        let mut rl2 = IdRangeList::new(v2.clone());

        rl1.sort();
        rl2.sort();

        (rl1, rl2)
    }

    fn prepare_rangesets(count1: u32, count2: u32) -> (IdRangeSet, IdRangeSet) {
        let (v1, v2) = prepare_vectors(count1, count2);
        (IdRangeSet::new(v1.clone()), IdRangeSet::new(v2.clone()))

    }

    const DEFAULT_COUNT: u32 = 100;

    #[bench]
    fn bench_rangelist_union_homo(b: &mut Bencher) {
        let (rl1, rl2) = prepare_rangelists(DEFAULT_COUNT, DEFAULT_COUNT);
        b.iter(|| {black_box(rl1.union(&rl2).sum::<u32>());});
    }

    #[bench]
    fn bench_rangeset_union_homo(b: &mut Bencher) {
        let (rl1, rl2) = prepare_rangesets(DEFAULT_COUNT, DEFAULT_COUNT);
        b.iter(|| {black_box(rl1.union(&rl2).sum::<u32>());});
    }

    #[bench]
    fn bench_rangelist_symdiff_homo(b: &mut Bencher) {
        let (rl1, rl2) = prepare_rangelists(DEFAULT_COUNT, DEFAULT_COUNT);
        b.iter(|| {black_box(rl1.symmetric_difference(&rl2).sum::<u32>());});
    }

    #[bench]
    fn bench_rangeset_symdiff_homo(b: &mut Bencher) {
        let (rl1, rl2) = prepare_rangesets(DEFAULT_COUNT, DEFAULT_COUNT);
        b.iter(|| {black_box(rl1.symmetric_difference(&rl2).sum::<u32>());});
    }

    #[bench]
    fn bench_rangelist_difference_homo(b: &mut Bencher) {
        let (rl1, rl2) = prepare_rangelists(DEFAULT_COUNT, DEFAULT_COUNT);
        b.iter(|| {black_box(rl1.difference(&rl2).sum::<u32>());});
    }

    #[bench]
    fn bench_rangeset_difference_homo(b: &mut Bencher) {
        let (rl1, rl2) = prepare_rangesets(DEFAULT_COUNT, DEFAULT_COUNT);
        b.iter(|| {black_box(rl1.difference(&rl2).sum::<u32>());});
    }

    #[bench]
    fn bench_rangelist_difference_hetero(b: &mut Bencher) {
        let (rl1, rl2) = prepare_rangelists(DEFAULT_COUNT, 10);
        b.iter(|| {black_box(rl1.difference(&rl2).sum::<u32>());});
    }

    #[bench]
    fn bench_rangeset_difference_hetero(b: &mut Bencher) {
        let (rl1, rl2) = prepare_rangesets(DEFAULT_COUNT, 10);
        b.iter(|| {black_box(rl1.difference(&rl2).sum::<u32>());});

    }

    #[bench]
    fn bench_rangelist_intersection(b: &mut Bencher) {
        let (rl1, rl2) = prepare_rangelists(DEFAULT_COUNT, DEFAULT_COUNT);
        b.iter(|| {black_box(rl1.intersection(&rl2).sum::<u32>());});
    }

    #[bench]
    fn bench_rangeset_intersection(b: &mut Bencher) {
        let (rl1, rl2) = prepare_rangesets(DEFAULT_COUNT, DEFAULT_COUNT);
        b.iter(|| {black_box(rl1.intersection(&rl2).sum::<u32>());});
    }

    #[bench]
    fn bench_rangelist_creation_shuffle(b: &mut Bencher) {
        let (v1, _) = prepare_vectors(DEFAULT_COUNT, DEFAULT_COUNT);
        b.iter(|| { let mut rl1 = IdRangeList::new(v1.clone());
                    rl1.sort();});
    }

    #[bench]
    fn bench_rangelist_creation_sorted(b: &mut Bencher) {
        let (mut v1, _) = prepare_vectors(DEFAULT_COUNT, DEFAULT_COUNT);
        v1.sort();
        b.iter(|| { let mut rl1 = IdRangeList::new(v1.clone());
                    rl1.sort();});
    }

    #[bench]
    fn bench_rangelist_creation_ranges(b: &mut Bencher) {
        let v1 = prepare_vector_ranges(100, 10);
        b.iter(|| { let mut rl1 = IdRangeList::new(v1.clone());
                    rl1.sort();});
    }

    #[bench]
    fn bench_rangeset_creation(b: &mut Bencher) {
        let (v1, _) = prepare_vectors(DEFAULT_COUNT, DEFAULT_COUNT);
        b.iter(|| { let _rs1 = IdRangeSet::new(v1.clone());});
    }

    #[bench]
    fn bench_rangeset_creation_sorted(b: &mut Bencher) {
        let (mut v1, _) = prepare_vectors(DEFAULT_COUNT, DEFAULT_COUNT);
        v1.sort();
        b.iter(|| { let _rs1 = IdRangeSet::new(v1.clone());});
    }

    #[bench]
    fn bench_rangeset_creation_ranges(b: &mut Bencher) {
        let v1 = prepare_vector_ranges(100, 10);
        b.iter(|| { let _rs1 = IdRangeSet::new(v1.clone());});
    }

    #[bench]
    fn bench_idset_intersection(b: &mut Bencher) {
        let mut id1: IdSet<IdRangeList> = IdSet::new();
        let mut id2: IdSet<IdRangeList> = IdSet::new();

        id1.push("node[0-1000000]");
        id2.push("node[1-1000001]");

        b.iter(|| { let _rs1 = id1.intersection(&id2);});
    }

    #[bench]
    fn bench_idset_intersection_set(b: &mut Bencher) {
        let mut id1: IdSet<IdRangeSet> = IdSet::new();
        let mut id2: IdSet<IdRangeSet> = IdSet::new();

        id1.push("node[0-1000000]");
        id2.push("node[1-1000001]");

        b.iter(|| { let _rs1 = id1.intersection(&id2);});
    }

    #[bench]
    fn bench_idset_print(b: &mut Bencher) {
        let mut id1: IdSet<IdRangeList> = IdSet::new();

        id1.push("node[0-10000000]");

        b.iter(|| {let _rs1 = id1.to_string();});
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_idset_parse() {
        let mut id1: IdSet<IdRangeList> = IdSet::new();
        let mut id2: IdSet<IdRangeList> = IdSet::new();

        id1.push("node[1-10/2,5][1-7]3,[1-10/2,5][1-7]2");
        id2.push("node[2-5][7][2,3]");

        /* ids.push("node0");
        println!("{:?}", ids);
        ids.push("node[1]");
        println!("{:?}", ids);
         */
        println!("{}", id1);
        println!("{}", id2);
        println!("{}", id1.intersection(&id2));

        assert!(false);
    }

    #[test]
    fn test_exponential_search() {
        assert_eq!(exponential_search(&vec![], &0), Err(0));
        assert_eq!(exponential_search(&vec![1, 2, 4, 7], &4), Ok(2));
        assert_eq!(exponential_search(&vec![1, 2, 4, 7], &0), Err(0));
        assert_eq!(exponential_search(&vec![1, 2, 4, 7], &8), Err(4));
    }

   fn validate_rangelist_union_result(a: Vec<u32>, b: Vec<u32>, c: Vec<u32>)
   {
        let rl1 = IdRangeList {
            indexes: a,
            sorted: true
        };

        let rl2 = IdRangeList {
            indexes: b,
            sorted: true
        };
        assert_eq!(rl1.union(&rl2).copied().collect::<Vec<u32>>(), c);
   }

   fn validate_rangelist_symdiff_result(a: Vec<u32>, b: Vec<u32>, c: Vec<u32>)
   {
        let rl1 = IdRangeList {
            indexes: a,
            sorted: true
        };

        let rl2 = IdRangeList {
            indexes: b,
            sorted: true
        };
        assert_eq!(rl1.symmetric_difference(&rl2).cloned().collect::<Vec<u32>>(), c);
   }

   fn validate_rangelist_intersection_result(a: Vec<u32>, b: Vec<u32>, c: Vec<u32>)
   {
        let rl1 = IdRangeList {
            indexes: a,
            sorted: true
        };

        let rl2 = IdRangeList {
            indexes: b,
            sorted: true
        };
        assert_eq!(rl1.intersection(&rl2).cloned().collect::<Vec<u32>>(), c);
   }
    #[test]
    fn rangelist_union() {
        validate_rangelist_union_result(vec![0,4,9], vec![1,2,5,7], vec![0,1,2,4,5,7,9]);
        validate_rangelist_union_result(vec![], vec![1,2,5,7], vec![1,2,5,7]);
        validate_rangelist_union_result(vec![0,4,9], vec![], vec![0,4,9]);
        validate_rangelist_union_result(vec![0,4,9], vec![10,11,12], vec![0,4,9,10,11,12]);
    }

    #[test]
    fn rangelist_symdiff() {
        validate_rangelist_symdiff_result(vec![0,2,4,7,9], vec![1,2,5,7], vec![0,1,4,5,9]);
        validate_rangelist_symdiff_result(vec![], vec![1,2,5,7], vec![1,2,5,7]);
        validate_rangelist_symdiff_result(vec![0,4,9], vec![], vec![0,4,9]);
        validate_rangelist_symdiff_result(vec![0,4,9], vec![10,11,12], vec![0,4,9,10,11,12]);
    }

    #[test]
    fn rangelist_intersection() {
        validate_rangelist_intersection_result(vec![0,4,9], vec![1,2,5,7], vec![]);
/*         validate_rangelist_intersection_result(vec![], vec![1,2,5,7], vec![]); */
        /* validate_rangelist_intersection_result(vec![0,4,9], vec![], vec![]);
        validate_rangelist_intersection_result(vec![0,4,9,7,12,34,35], vec![4,11,12,37], vec![4,12]);
        validate_rangelist_intersection_result(vec![4,11,12,37], vec![0,4,9,7,12,34,35], vec![4,12]); */
    }

    #[test]
    fn rangelist_difference() {
        let rl1 = IdRangeList {
            indexes: vec![1, 2, 3],
            sorted: true
        };

        let mut rl2 = IdRangeList {
            indexes: vec![1, 3],
            sorted: true
        };
        assert_eq!(rl1.difference(&rl2).cloned().collect::<Vec<u32>>(), vec![2]);

        rl2 = IdRangeList {
            indexes: vec![],
            sorted: true
        };
        assert_eq!(rl1.difference(&rl2).cloned().collect::<Vec<u32>>(), vec![1, 2, 3]);
        assert_eq!(rl2.difference(&rl1).cloned().collect::<Vec<u32>>(), vec![]);
        rl2 = IdRangeList {
            indexes: vec![4,5,6],
            sorted: true
        };
        assert_eq!(rl1.difference(&rl2).cloned().collect::<Vec<u32>>(), vec![1, 2, 3]);
        assert_eq!(rl1.difference(&rl1).cloned().collect::<Vec<u32>>(), vec![]);
    }
    #[test]
    #[should_panic]
    fn rangelist_difference_bad1(){
        let rl1 = IdRangeList {
            indexes: vec![1, 2, 3],
            sorted: true
        };

        let rl2 = IdRangeList {
            indexes: vec![1, 3],
            sorted: false
        };

        rl1.difference(&rl2);
    }
}


/* type Coord1 = u32;
type Coord2 = (u32, u32);
type Coord3 = (u32, u32, u32);
type Coord4 = (u32, u32, u32, u32);

impl Coord for Coord1 {
    fn initcoords(self: Self) -> AnyCoords {
        let v = vec![self];
        AnyCoords::Coords1(v)
    }
    fn addto(self: Self, coords: &mut AnyCoords){
        match coords {
            AnyCoords::Coords1(c) => c.push(self),
            _ => panic!("Coord type mismatch")
        }
    }
}
 */

/*
enum AnyCoords {
    Coords1(Vec<Coord1>),
    Coords2(Vec<Coord2>),
    Coords3(Vec<Coord3>),
    Coords4(Vec<Coord4>),
}
 */

/*
    push_one(cmpnts, &u[32])
      id = get_nodeid(cmpnts)
      if last.nodeid == id
        last.push(&[u32])
      else
        rngs.push(nr::new(&[u32]))

    iter(self, nodelist)
        self.rngiter = nodelist.rngs.iter()
        self.nriter = slef.rngiter.next()

        coord = nriter().next_or(self.rngiter.next())
        return format cpmnts[nodeid][]

      for i in rngs()
         retur
    nr.push(self, &[u32])
    nr.append(self, other)
    nr.intersect(self, other)
    nr.exclude(self, other)
    nr.iter()
    nr.iterfold()
    nr.nodeid()
*/

// Algo:

// pour n[0-9]m[10-11]

// stocker:
// names: ['n','m']
// vec: (0,10), (0,11), (1,10) ...

// dans un map indexé par les names
// puis: trier et dedupliquer vec ordre lexico
// puis: parcourir vec et recréer un nouveau vec: tant que seule la derniere dimension change de +1 stocker debut fin
// generer nouveau vec (0, (10,11)), (1,(10,11)) etc.
// puis recommencer avec l'avant dernière dimension (trier à nouveau mais en changeant le lexicographique: dimension n-1 en dernier) etc.
