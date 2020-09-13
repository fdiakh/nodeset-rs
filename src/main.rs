#![cfg_attr(feature = "unstable", feature(test))]

use fnv::{FnvBuildHasher, FnvHashSet};
use itertools::Itertools;
use rand::prelude::*;
use regex::Regex;
use std::collections::hash_set::{
    Difference as HSDifference, Intersection as HSIntersection,
    SymmetricDifference as HSSymmetricDifference, Union as HSUnion,
};
use std::collections::HashMap;
use std::io;

//todo: implement partialEq wrt sorted
#[derive(Debug, PartialEq, Clone)]
pub struct IdRangeList {
    indexes: Vec<u32>,
    sorted: bool,
}

#[derive(Debug, PartialEq, Clone)]
pub struct IdRangeSet {
    indexes: FnvHashSet<u32>,
}

struct VecDifference<'a, T> {
    a: std::slice::Iter<'a, T>,
    b: std::iter::Peekable<std::slice::Iter<'a, T>>,
}

struct VecIntersection<'a, T> {
    a: &'a [T],
    b: &'a [T],
}

struct VecUnion<'a, T> {
    a: &'a [T],
    b: &'a [T],
}

struct VecSymDifference<'a, T> {
    a: &'a [T],
    b: &'a [T],
}

impl IdRangeSet {}

impl IdRangeList {
    fn sort(self: &mut Self) {
        self.indexes.sort_unstable();
        self.sorted = true;
    }
}

impl<'a, T> Iterator for VecDifference<'a, T>
where
    T: Ord + std::fmt::Debug,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let mut next = match self.a.next() {
            Some(v) => v,
            None => return None,
        };

        let mut min: &T;
        loop {
            min = match self.b.peek() {
                None => return Some(next),
                Some(v) if *v == next => v,
                Some(v) if *v > next => return Some(next),
                _ => {
                    self.b.next();
                    continue;
                }
            };

            while next == min {
                next = match self.a.next() {
                    Some(v) => v,
                    None => return None,
                };
            }
        }
    }
}

impl<'a, T> Iterator for VecIntersection<'a, T>
where
    T: Ord + std::fmt::Debug,
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
            } else if first_a < first_b {
                self.a = &self.a[exponential_search_idx(self.a, &first_b.unwrap())..];
            } else {
                self.b = &self.b[exponential_search_idx(self.b, &first_a.unwrap())..];
            }
        }

        return None;
    }
}

impl<'a, T> Iterator for VecUnion<'a, T>
where
    T: Ord + std::fmt::Debug,
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
where
    T: Ord + std::fmt::Debug,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let mut first_a = self.a.first();
        let mut first_b = self.b.first();

        while first_a == first_b {
            if first_a == None {
                return None;
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
    type SelfIter: Iterator<Item = &'a u32> + Clone;
    type DifferenceIter: Iterator<Item = &'a u32>;
    type SymmetricDifferenceIter: Iterator<Item = &'a u32>;
    type IntersectionIter: Iterator<Item = &'a u32>;
    type UnionIter: Iterator<Item = &'a u32>;

    fn new(indexes: Vec<u32>) -> Self;
    fn difference(self: &'a Self, other: &'a Self) -> Self::DifferenceIter;
    fn symmetric_difference(self: &'a Self, other: &'a Self) -> Self::SymmetricDifferenceIter;
    fn intersection(self: &'a Self, other: &'a Self) -> Self::IntersectionIter;
    fn union(self: &'a Self, other: &'a Self) -> Self::UnionIter;
    fn iter(self: &'a Self) -> Self::SelfIter;
    fn contains(self: &Self, id: u32) -> bool;
    fn is_empty(self: &Self) -> bool;
    fn push(self: &mut Self, other: &Self);
    fn len(self: &Self) -> usize;
}

impl<'a> IdRange<'a> for IdRangeSet {
    type DifferenceIter = HSDifference<'a, u32, FnvBuildHasher>;
    type SymmetricDifferenceIter = HSSymmetricDifference<'a, u32, FnvBuildHasher>;
    type IntersectionIter = HSIntersection<'a, u32, FnvBuildHasher>;
    type UnionIter = HSUnion<'a, u32, FnvBuildHasher>;
    type SelfIter = std::collections::hash_set::Iter<'a, u32>;

    fn len(self: &Self) -> usize {
        self.indexes.len()
    }

    fn push(self: &mut Self, other: &Self){
        self.indexes.extend(&other.indexes);
    }

    fn new(indexes: Vec<u32>) -> IdRangeSet {
        let mut bt: FnvHashSet<u32> = FnvHashSet::with_hasher(Default::default());
        bt.extend(&indexes);
        IdRangeSet { indexes: bt }
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
        return self.indexes.is_empty();
    }
    fn iter(self: &'a Self) -> Self::SelfIter {
        return self.indexes.iter();
    }
}

fn exponential_search<T>(v: &[T], x: &T) -> Result<usize, usize>
where
    T: Ord,
{
    let mut i: usize = 1;
    while i <= v.len() {
        if v[i - 1] == *x {
            return Ok(i - 1);
        }
        if v[i - 1] > *x {
            break;
        }
        i *= 2;
    }

    match v[i / 2..std::cmp::min(i - 1, v.len())].binary_search(x) {
        Ok(result) => Ok(result + i / 2),
        Err(result) => Err(result + i / 2),
    }
}

fn exponential_search_idx<T>(v: &[T], x: &T) -> usize
where
    T: Ord,
{
    match exponential_search(v, x) {
        Ok(r) => r,
        Err(r) => r,
    }
}

impl<'a> IdRange<'a> for IdRangeList {
    type DifferenceIter = VecDifference<'a, u32>;
    type IntersectionIter = VecIntersection<'a, u32>;
    type UnionIter = VecUnion<'a, u32>;
    type SymmetricDifferenceIter = VecSymDifference<'a, u32>;
    type SelfIter = std::slice::Iter<'a, u32>;
    fn new(indexes: Vec<u32>) -> IdRangeList {
        IdRangeList {
            indexes,
            sorted: false,
        }
    }

    fn len(self: &Self) -> usize {
        self.indexes.len()
    }

    fn push(self: &mut Self, other: &Self){
        let need_sort = other.indexes[0] < *self.indexes.last().unwrap_or(&0);

        self.indexes.extend(other.iter());
        if need_sort {
            self.indexes.sort_unstable();
        }
    }

    fn contains(self: &Self, id: u32) -> bool {
        exponential_search(&self.indexes, &id).is_ok()
    }

    fn iter(self: &'a Self) -> Self::SelfIter {
        self.indexes.iter()
    }

    fn intersection(self: &'a Self, other: &'a Self) -> Self::IntersectionIter {
        Self::IntersectionIter {
            a: &self.indexes,
            b: &other.indexes,
        }
    }

    fn union(self: &'a Self, other: &'a Self) -> Self::UnionIter {
        Self::UnionIter {
            a: &self.indexes,
            b: &other.indexes,
        }
    }

    fn symmetric_difference(self: &'a Self, other: &'a Self) -> Self::SymmetricDifferenceIter {
        Self::SymmetricDifferenceIter {
            a: &self.indexes,
            b: &other.indexes,
        }
    }

    fn difference(self: &'a Self, other: &'a Self) -> Self::DifferenceIter {
        //TODO: fix this once we manage sorted
        //assert!(self.sorted);
        //assert!(other.sorted);
        Self::DifferenceIter {
            a: self.indexes.iter(),
            b: other.indexes.iter().peekable(),
        }
    }

    fn is_empty(self: &Self) -> bool {
        return self.indexes.is_empty();
    }
}

#[derive(Debug, Clone, PartialEq)]
struct IdRangeProduct<T> {
    ranges: Vec<T>,
}

impl<'a, T> IdRangeProduct<T>
where
    T: IdRange<'a>,
{
    fn intersection(self: &'a Self, other: &'a Self) -> Option<IdRangeProduct<T>> {
        let mut ranges = Vec::<T>::new();

        for (sidr, oidr) in self.ranges.iter().zip(other.ranges.iter()) {
            let rng = T::new(sidr.intersection(&oidr).cloned().collect::<Vec<u32>>());
            if rng.is_empty() {
                return None;
            }
            ranges.push(rng);
        }

        Some(IdRangeProduct { ranges })
    }

    fn iter(self: &Self) -> std::slice::Iter<T> {
        self.ranges.iter()
    }
}

#[derive(Debug, PartialEq)]
struct IdSet<T> {
    products: Vec<IdRangeProduct<T>>,
}

impl<T> std::fmt::Display for IdSet<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // Customize so only `x` and `y` are denoted.
        write!(f, "{}", self.products.iter().join(","))
    }
}

impl<T> std::fmt::Display for IdRangeProduct<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // Customize so only `x` and `y` are denoted.
        write!(f, "{}", self.ranges.iter().join(""))
    }
}

impl std::fmt::Display for IdRangeList {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut rngs = self
            .indexes
            .iter()
            .zip(
                self.indexes
                    .iter()
                    .chain(self.indexes.iter().last())
                    .skip(1),
            )
            .batching(|it| {
                if let Some((&first, &next)) = it.next() {
                    if next != first + 1 {
                        return Some(first.to_string());
                    }
                    for (&cur, &next) in it {
                        if next != cur + 1 {
                            return Some(format!("{}-{}", first, cur));
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

impl std::fmt::Display for IdRangeSet {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "[{}]", self.indexes.iter().join(","))
    }
}

impl<T> IdSet<T>
where
    for<'a> T: IdRange<'a> + PartialEq + Clone + std::fmt::Display + std::fmt::Debug,
{
    // TODO: return Result instead of panics
    // TODO: Enforce dimname for first match for nodeset
    fn push(self: &mut Self, nodelist: &str) {
        //let re = Regex::new(r"\s*(?P<dimname>@?[a-zA-Z_][a-zA-Z_\.\-]*)(?:(?P<index>[0-9]+)|(?:\[(?P<sindex>[0-9]+)(?:-(?P<eindex>[0-9]+))?(?:/(?P<step>[0-9]+))?\]))?(?P<sep>[,])?").unwrap();
        let dim_re = Regex::new(r"\s*(?P<dimname>@?[a-zA-Z_][a-zA-Z_\.\-]*)?(?:(?P<index>[0-9]+)|(?:\[(?P<range>[0-9,-/]+)\]))?(?P<sep>[,])?").unwrap();
        let range_re = Regex::new(
            r"(?P<sindex>[0-9]+)(?:-(?P<eindex>[0-9]+))?(?:/(?P<step>[0-9]+))?(?P<sep>[,])?",
        )
        .unwrap();
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
                        panic!(
                            "Failed to parse at char {}: {}",
                            prev_index,
                            &nodelist[prev_index..new_index]
                        );
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
                        range.extend((sindex..eindex + 1).step_by(step))
                    } else if let Some(m) = rng_caps.name("sindex") {
                        // println!("single");
                        range.push(m.as_str().parse().unwrap())
                    } else {
                        panic!(
                            "Failed to parse at char {}: {}",
                            prev_index,
                            &nodelist[prev_index..new_index]
                        );
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
                self.products.push(IdRangeProduct { ranges });
                ranges = Vec::<T>::new();
            }
            prev_index = caps.get(0).unwrap().end();
        }

        if !ranges.is_empty() {
            self.products.push(IdRangeProduct { ranges });
        }
    }
    fn sort(self: &mut Self, skip: usize) {
        self.products[skip..].sort_unstable_by(|a, b| {
            for (ai, bi) in a.iter().zip(b.iter()) {
                let ai0 = ai.iter().next();
                let bi0 = bi.iter().next();

                match ai0.cmp(&bi0) {
                    std::cmp::Ordering::Less => return std::cmp::Ordering::Less,
                    std::cmp::Ordering::Greater => return std::cmp::Ordering::Greater,
                    _ => continue,
                }
            }
            return std::cmp::Ordering::Equal;
        });
    }
    fn full_split(self: &mut Self) {
       /*  println!("full split"); */
        let mut split_products = vec![];
        for p in &self.products {
            // Each product is split into a list of products
            let mut first_range = true;
            let start = split_products.len();
            for rng in &p.ranges {
                if first_range {
                    first_range = false;
                    for n in rng.iter() {
                        split_products.push(IdRangeProduct{ranges: vec![T::new(vec![*n])]});
                    }
                } else {
                    let count = split_products.len() - start;
                    for _ in 1..rng.len(){
                        for j in 0..count {
                            split_products.push(split_products[start + j].clone())
                        }
                    }

                    for (i, r) in rng.iter().enumerate() {
                        for sp in split_products[start+i*count..start+(i+1)*count].iter_mut() {
                            sp.ranges.push(T::new(vec![*r]));
                        }
                    }
                }
            }
        }
        self.products = split_products;
        /* println!("splitted"); */
        self.sort(0);
        self.products.dedup();
        /* println!("sorted"); */
    }

    fn minimal_split(self: &mut Self) {
        /* println!("minimal split"); */
        self.sort(0);
        /* println!("sorted"); */
        let mut idx1 = 0;
        while idx1 < self.products.len() - 1{
            let mut cur_len = self.products.len();
            let mut idx2 = idx1 + 1;
            while idx2 < cur_len {
                let p1 = &self.products[idx1];
                let p2 = &self.products[idx2];
  /*            println!("compare p1 {} with p2 {}", p1, p2); */
                let mut inter_p = IdRangeProduct::<T>{ranges: vec![]};
                let mut new_products = vec![];
                let mut keep = false;
                let mut term = false;
                let mut intersect = false;
                let mut split = false;
                // todo: define len on products
                let num_axis = p1.ranges.len();
                for (axis, (r1, r2)) in p1.iter()
                                        .zip(p2.iter())
                                        .enumerate() {

                    intersect = r2.intersection(r1).next().is_some();
                    if !intersect && ( axis < num_axis - 1 || ! split) {
                        keep = true;
                        if p1.ranges[0].iter().last() < p2.ranges[0].iter().next(){
                            term = true;
                        }
                        break;
                    }

                    if !intersect {
                        //println!("split push");
                        new_products.push(IdRangeProduct{ranges: inter_p.ranges[0..axis].iter()
                                                                .chain(&(p1.ranges[axis..]))
                                                                .cloned()
                                                                .collect()});
                        new_products.push(IdRangeProduct{ranges: inter_p.ranges[0..axis].iter()
                                                                .chain(&(p2.ranges[axis..]))
                                                                .cloned()
                                                                .collect()});
                    } else if r1 == r2 {
                        //println!("same range {} {}", r1, r2);
                        inter_p.ranges.push(r1.clone());
                    }  else {
                        //println!("split range");
                        // todo: keep intersect iter
                        split = true;
                        inter_p.ranges.push(T::new(r1.intersection(r2).cloned().collect()));
                        if r1.difference(r2).next().is_some() {
                            let diff = vec![T::new(r1.difference(r2).cloned().collect())];
                            //println!("diff1 {:?}", diff);
                            let new_iter = inter_p.ranges[0..axis].iter()
                                                             .chain(&diff)
                                                             .chain(&(p1.ranges[axis+1..]));
                            new_products.push(IdRangeProduct{ranges: new_iter.cloned().collect()});
                        }
                        if r2.difference(r1).next().is_some() {
                            let diff = vec![T::new(r2.difference(r1).cloned().collect())];
                            //println!("diff2 {:?}", diff);
                            let new_iter = inter_p.ranges[0..axis].iter()
                                                             .chain(&diff)
                                                             .chain(&(p2.ranges[axis+1..]));
                            new_products.push(IdRangeProduct{ranges: new_iter.cloned().collect()});
                        }
                    }
                }
                if !keep{
                    if intersect {
                        //println!("intersect");
                      self.products[idx1] = inter_p;
                    } else {
                        //println!("no intersect");
                        self.products[idx1] = new_products[0].clone();
                        new_products.swap_remove(0);
                    }
                    if idx2 < cur_len - 1 {
                        //println!("idx2 not at end");
                        self.products.swap(idx2, cur_len - 1);
                    }
                    self.products.swap_remove(cur_len - 1);
                    cur_len -= 1;
                    //println!("inserting {}", new_products.len());
                    self.products.append(&mut new_products);
                }else{
                    //println!("keep");
                    idx2+=1;
                }
                if term {
                    //println!("term");
                    break;
                }
            }
            self.sort(idx1);
            //println!("next p1");
            idx1+=1
        }
    }

    fn merge(self: &mut Self){
        /* println!("merge"); */
        let mut dellst = vec![false; self.products.len()];

        loop {
            /* println!("loop"); */
            let mut update = false;
            let mut idx1 = 0;
            while idx1 < self.products.len() - 1 {
                if dellst[idx1]{
                    idx1 += 1;
                    continue
                }
                let mut idx2 = idx1 + 1;
                while idx2 < self.products.len() {
                    if dellst[idx2]{
                        idx2 += 1;
                        continue
                    }


                    //let mut new_p = vec![];
                    let mut num_diffs = 0;
                    let mut range = 0;
                    {

                    let p2 = &self.products[idx2];
                    let p1 = &self.products[idx1];
/*                      println!("try merge p1 {} with p2 {}", p1, p2); */
                    for (i, (r1, r2)) in p1.iter().zip(p2.iter()).enumerate() {
                        if r1 == r2 {
                            /* println!("same range"); */
                        } else if num_diffs == 0 {
                            /* println!("merge range"); */
                            //let new_rng = T::new(r1.union(r2).cloned().collect());
                            //new_p.push(new_rng);
                            // todo: store which range
                            // todo: append instead of union
                            num_diffs += 1;
                            range = i;
                        } else {
                            /* println!("abort"); */
                            num_diffs += 1;
                            break;
                        }
                    }
                    }
                    if num_diffs < 2 {
                        //println!("merge product");
                        update = true;
                        let (pp1, pp2) = self.products.split_at_mut(idx2);

                        pp1[idx1].ranges[range].push(&pp2[0].ranges[range]);

                        dellst[idx2] = true;
                        //self.products.swap_remove(idx2);
                    }/*  else {
                        //println!("next product"); */
                        idx2 += 1
                   /*  } */
                }
                idx1 += 1;
            }
            if !update {
                    break
            }
        }
        self.products = self.products
        .iter()
        .cloned()
        .enumerate()
        .filter(|(i, _)| {!dellst[*i]})
        .map(|(_, p)| {p})
        .collect();
    }

    fn fold(self: &mut Self) {
        /* self.minimal_split();*/
        self.full_split();
        self.merge();
    }
    fn intersection(self: &Self, other: &Self) -> Self {
        let mut products = Vec::<IdRangeProduct<T>>::new();
        for (sidpr, oidpr) in self
            .products
            .iter()
            .cartesian_product(other.products.iter())
        {
            if let Some(idpr) = sidpr.intersection(oidpr) {
                products.push(idpr)
            }
        }
        IdSet { products }
    }

    fn new() -> Self {
        IdSet {
            products: Vec::new(),
        }
    }
}

pub struct NodeSet<T> {
    nodenames: HashMap<String, IdSet<T>>,
}

fn main() {
    let mut text = String::new();
    io::stdin()
        .read_line(&mut text)
        .expect("Failed to read line");

    let mut r = IdSet::<IdRangeList>::new();

    r.push(&text);
    r.fold();
    println!("{}", r);
}

#[cfg(all(feature = "unstable", test))]
mod benchs {
    extern crate test;
    use super::*;
    use test::{black_box, Bencher};

    fn prepare_vector_ranges(count: u32, ranges: u32) -> Vec<u32> {
        let mut res: Vec<u32> = Vec::new();
        for i in (0..ranges).rev() {
            res.append(&mut (count * i..count * (i + 1)).collect());
        }
        return res;
    }

    fn prepare_vectors(count1: u32, count2: u32) -> (Vec<u32>, Vec<u32>) {
        let mut v1: Vec<u32> = (0..count1).collect();
        let mut v2: Vec<u32> = (1..count2 + 1).collect();
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
        b.iter(|| {
            black_box(rl1.union(&rl2).sum::<u32>());
        });
    }

    #[bench]
    fn bench_rangeset_union_homo(b: &mut Bencher) {
        let (rl1, rl2) = prepare_rangesets(DEFAULT_COUNT, DEFAULT_COUNT);
        b.iter(|| {
            black_box(rl1.union(&rl2).sum::<u32>());
        });
    }

    #[bench]
    fn bench_rangelist_symdiff_homo(b: &mut Bencher) {
        let (rl1, rl2) = prepare_rangelists(DEFAULT_COUNT, DEFAULT_COUNT);
        b.iter(|| {
            black_box(rl1.symmetric_difference(&rl2).sum::<u32>());
        });
    }

    #[bench]
    fn bench_rangeset_symdiff_homo(b: &mut Bencher) {
        let (rl1, rl2) = prepare_rangesets(DEFAULT_COUNT, DEFAULT_COUNT);
        b.iter(|| {
            black_box(rl1.symmetric_difference(&rl2).sum::<u32>());
        });
    }

    #[bench]
    fn bench_rangelist_difference_homo(b: &mut Bencher) {
        let (rl1, rl2) = prepare_rangelists(DEFAULT_COUNT, DEFAULT_COUNT);
        b.iter(|| {
            black_box(rl1.difference(&rl2).sum::<u32>());
        });
    }

    #[bench]
    fn bench_rangeset_difference_homo(b: &mut Bencher) {
        let (rl1, rl2) = prepare_rangesets(DEFAULT_COUNT, DEFAULT_COUNT);
        b.iter(|| {
            black_box(rl1.difference(&rl2).sum::<u32>());
        });
    }

    #[bench]
    fn bench_rangelist_difference_hetero(b: &mut Bencher) {
        let (rl1, rl2) = prepare_rangelists(DEFAULT_COUNT, 10);
        b.iter(|| {
            black_box(rl1.difference(&rl2).sum::<u32>());
        });
    }

    #[bench]
    fn bench_rangeset_difference_hetero(b: &mut Bencher) {
        let (rl1, rl2) = prepare_rangesets(DEFAULT_COUNT, 10);
        b.iter(|| {
            black_box(rl1.difference(&rl2).sum::<u32>());
        });
    }

    #[bench]
    fn bench_rangelist_intersection(b: &mut Bencher) {
        let (rl1, rl2) = prepare_rangelists(DEFAULT_COUNT, DEFAULT_COUNT);
        b.iter(|| {
            black_box(rl1.intersection(&rl2).sum::<u32>());
        });
    }

    #[bench]
    fn bench_rangeset_intersection(b: &mut Bencher) {
        let (rl1, rl2) = prepare_rangesets(DEFAULT_COUNT, DEFAULT_COUNT);
        b.iter(|| {
            black_box(rl1.intersection(&rl2).sum::<u32>());
        });
    }

    #[bench]
    fn bench_rangelist_creation_shuffle(b: &mut Bencher) {
        let (v1, _) = prepare_vectors(DEFAULT_COUNT*100, DEFAULT_COUNT*100);
        b.iter(|| {
            let mut rl1 = IdRangeList::new(v1.clone());
            rl1.sort();
        });
    }

    #[bench]
    fn bench_rangelist_creation_sorted(b: &mut Bencher) {
        let (mut v1, _) = prepare_vectors(DEFAULT_COUNT, DEFAULT_COUNT);
        v1.sort();
        b.iter(|| {
            let mut rl1 = IdRangeList::new(v1.clone());
            rl1.sort();
        });
    }

    #[bench]
    fn bench_rangelist_creation_ranges(b: &mut Bencher) {
        let v1 = prepare_vector_ranges(100, 10);
        b.iter(|| {
            let mut rl1 = IdRangeList::new(v1.clone());
            rl1.sort();
        });
    }

    #[bench]
    fn bench_rangeset_creation(b: &mut Bencher) {
        let (v1, _) = prepare_vectors(DEFAULT_COUNT, DEFAULT_COUNT);
        b.iter(|| {
            let _rs1 = IdRangeSet::new(v1.clone());
        });
    }

    #[bench]
    fn bench_rangeset_creation_sorted(b: &mut Bencher) {
        let (mut v1, _) = prepare_vectors(DEFAULT_COUNT, DEFAULT_COUNT);
        v1.sort();
        b.iter(|| {
            let _rs1 = IdRangeSet::new(v1.clone());
        });
    }

    #[bench]
    fn bench_rangeset_creation_ranges(b: &mut Bencher) {
        let v1 = prepare_vector_ranges(100, 10);
        b.iter(|| {
            let _rs1 = IdRangeSet::new(v1.clone());
        });
    }

    #[bench]
    fn bench_idset_intersection(b: &mut Bencher) {
        let mut id1: IdSet<IdRangeList> = IdSet::new();
        let mut id2: IdSet<IdRangeList> = IdSet::new();

        id1.push("node[0-1000000]");
        id2.push("node[1-1000001]");

        b.iter(|| {
            let _rs1 = id1.intersection(&id2);
        });
    }

    #[bench]
    fn bench_idset_intersection_set(b: &mut Bencher) {
        let mut id1: IdSet<IdRangeSet> = IdSet::new();
        let mut id2: IdSet<IdRangeSet> = IdSet::new();

        id1.push("node[0-1000000]");
        id2.push("node[1-1000001]");

        b.iter(|| {
            let _rs1 = id1.intersection(&id2);
        });
    }

    #[bench]
    fn bench_idset_print(b: &mut Bencher) {
        let mut id1: IdSet<IdRangeList> = IdSet::new();

        id1.push("node[0-10000000]");

        b.iter(|| {
            let _rs1 = id1.to_string();
        });
    }

    #[bench]
    fn bench_idset_split(b: &mut Bencher) {
        b.iter(|| {
            let mut id1: IdSet<IdRangeList> = IdSet::new();
            id1.push("node[0-100000]");
            id1.full_split();
        });
    }

    #[bench]
    fn bench_idset_split_set(b: &mut Bencher) {
        b.iter(|| {
            let mut id1: IdSet<IdRangeSet> = IdSet::new();
            id1.push("node[0-100000]");
            id1.full_split();
        });
    }

    #[bench]
    fn bench_idset_merge(b: &mut Bencher) {
        b.iter(|| {
            let mut id1: IdSet<IdRangeList> = IdSet::new();
            id1.push("node[0-100000]");
            id1.full_split();
            id1.merge();
        });
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

        assert_eq!(
            id1.to_string(),
            "[1,3,5,7,9,5][1-7][3],[1,3,5,7,9,5][1-7][2]"
        );
        assert_eq!(id2.to_string(), "[2-5][7][2-3]");
        assert_eq!(
            id1.intersection(&id2).to_string(),
            "[3,5][7][3],[3,5][7][2]"
        );
    }

    #[test]
    fn test_idset_fold() {
        let mut id1: IdSet<IdRangeList> = IdSet::new();
        let mut id2: IdSet<IdRangeList> = IdSet::new();
        let mut id3: IdSet<IdRangeList> = IdSet::new();

        id1.push("[1-10/2,5][1-7]3,[1-10/2,5][1-7]2");
        id2.push("[0-10][0-10],[0-20][0-10]");
        id3.push("[0-10]y[0-10],x[8-18]y[8-18],x[11-18]y[0-7]");
        id1.fold();
        id2.fold();
        id3.fold();

        assert_eq!(id1.to_string(), "[1,3,5,7,9][1-7][2-3]");
        assert_eq!(id2.to_string(), "[0-20][0-10]");
        assert_eq!(id3.to_string(), "[0-7][0-10],[8-18][0-18]");
    }

    #[test]
    fn test_exponential_search() {
        assert_eq!(exponential_search(&vec![], &0), Err(0));
        assert_eq!(exponential_search(&vec![1, 2, 4, 7], &4), Ok(2));
        assert_eq!(exponential_search(&vec![1, 2, 4, 7], &0), Err(0));
        assert_eq!(exponential_search(&vec![1, 2, 4, 7], &8), Err(4));
    }

    fn validate_rangelist_union_result(a: Vec<u32>, b: Vec<u32>, c: Vec<u32>) {
        let rl1 = IdRangeList {
            indexes: a,
            sorted: true,
        };

        let rl2 = IdRangeList {
            indexes: b,
            sorted: true,
        };
        assert_eq!(rl1.union(&rl2).copied().collect::<Vec<u32>>(), c);
    }

    fn validate_rangelist_symdiff_result(a: Vec<u32>, b: Vec<u32>, c: Vec<u32>) {
        let rl1 = IdRangeList {
            indexes: a,
            sorted: true,
        };

        let rl2 = IdRangeList {
            indexes: b,
            sorted: true,
        };
        assert_eq!(
            rl1.symmetric_difference(&rl2)
                .cloned()
                .collect::<Vec<u32>>(),
            c
        );
    }

    fn validate_rangelist_intersection_result(a: Vec<u32>, b: Vec<u32>, c: Vec<u32>) {
        let rl1 = IdRangeList {
            indexes: a,
            sorted: true,
        };

        let rl2 = IdRangeList {
            indexes: b,
            sorted: true,
        };
        assert_eq!(rl1.intersection(&rl2).cloned().collect::<Vec<u32>>(), c);
    }
    #[test]
    fn rangelist_union() {
        validate_rangelist_union_result(vec![0, 4, 9], vec![1, 2, 5, 7], vec![0, 1, 2, 4, 5, 7, 9]);
        validate_rangelist_union_result(vec![], vec![1, 2, 5, 7], vec![1, 2, 5, 7]);
        validate_rangelist_union_result(vec![0, 4, 9], vec![], vec![0, 4, 9]);
        validate_rangelist_union_result(vec![0, 4, 9], vec![10, 11, 12], vec![0, 4, 9, 10, 11, 12]);
    }

    #[test]
    fn rangelist_symdiff() {
        validate_rangelist_symdiff_result(
            vec![0, 2, 4, 7, 9],
            vec![1, 2, 5, 7],
            vec![0, 1, 4, 5, 9],
        );
        validate_rangelist_symdiff_result(vec![], vec![1, 2, 5, 7], vec![1, 2, 5, 7]);
        validate_rangelist_symdiff_result(vec![0, 4, 9], vec![], vec![0, 4, 9]);
        validate_rangelist_symdiff_result(
            vec![0, 4, 9],
            vec![10, 11, 12],
            vec![0, 4, 9, 10, 11, 12],
        );
    }

    #[test]
    fn rangelist_intersection() {
        validate_rangelist_intersection_result(vec![0, 4, 9], vec![1, 2, 5, 7], vec![]);
        /*         validate_rangelist_intersection_result(vec![], vec![1,2,5,7], vec![]); */
        /* validate_rangelist_intersection_result(vec![0,4,9], vec![], vec![]);
        validate_rangelist_intersection_result(vec![0,4,9,7,12,34,35], vec![4,11,12,37], vec![4,12]);
        validate_rangelist_intersection_result(vec![4,11,12,37], vec![0,4,9,7,12,34,35], vec![4,12]); */
    }

    #[test]
    fn rangelist_difference() {
        let rl1 = IdRangeList {
            indexes: vec![1, 2, 3],
            sorted: true,
        };

        let mut rl2 = IdRangeList {
            indexes: vec![1, 3],
            sorted: true,
        };
        assert_eq!(rl1.difference(&rl2).cloned().collect::<Vec<u32>>(), vec![2]);

        rl2 = IdRangeList {
            indexes: vec![],
            sorted: true,
        };
        assert_eq!(
            rl1.difference(&rl2).cloned().collect::<Vec<u32>>(),
            vec![1, 2, 3]
        );
        assert_eq!(rl2.difference(&rl1).cloned().collect::<Vec<u32>>(), vec![]);
        rl2 = IdRangeList {
            indexes: vec![4, 5, 6],
            sorted: true,
        };
        assert_eq!(
            rl1.difference(&rl2).cloned().collect::<Vec<u32>>(),
            vec![1, 2, 3]
        );
        assert_eq!(rl1.difference(&rl1).cloned().collect::<Vec<u32>>(), vec![]);
    }
    // TODO: once we restore sorted test
    /* #[test]
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
    } */
}
