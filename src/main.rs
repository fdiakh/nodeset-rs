#![cfg_attr(feature = "unstable", feature(test))]

use clap::{App, Arg, SubCommand};
use itertools::Itertools;
use std::collections::btree_set;
use std::collections::{BTreeSet, HashMap};
use std::fmt;

#[derive(Debug, Clone)]
pub struct IdRangeList {
    indexes: Vec<u32>,
    sorted: bool,
}

impl PartialEq for IdRangeList {
    fn eq(&self, other: &Self) -> bool {
        self.indexes == other.indexes
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct IdRangeTree {
    indexes: BTreeSet<u32>,
}

pub struct VecDifference<'a, T> {
    a: std::slice::Iter<'a, T>,
    b: std::iter::Peekable<std::slice::Iter<'a, T>>,
}

pub struct VecIntersection<'a, T> {
    a: &'a [T],
    b: &'a [T],
}

pub struct VecUnion<'a, T> {
    a: &'a [T],
    b: &'a [T],
}

pub struct VecSymDifference<'a, T> {
    a: &'a [T],
    b: &'a [T],
}

impl IdRangeList {}

impl<'a, T> Iterator for VecDifference<'a, T>
where
    T: Ord + fmt::Debug,
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
    T: Ord + fmt::Debug,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        while !self.a.is_empty() && !self.b.is_empty() {
            let first_a = self.a.first();
            let first_b = self.b.first();

            match first_a.cmp(&first_b) {
                std::cmp::Ordering::Less => {
                    self.a = &self.a[exponential_search_idx(self.a, first_b.unwrap())..];
                }

                std::cmp::Ordering::Equal => {
                    self.a = &self.a[1..];
                    self.b = &self.b[1..];
                    return first_a;
                }
                std::cmp::Ordering::Greater => {
                    self.b = &self.b[exponential_search_idx(self.b, first_a.unwrap())..];
                }
            }
        }

        None
    }
}

impl<'a, T> Iterator for VecUnion<'a, T>
where
    T: Ord + fmt::Debug,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let first_a = self.a.first();
        let first_b = self.b.first();

        if first_a.is_some() && (first_a <= first_b || first_b.is_none()) {
            self.a = &self.a[1..];
            return first_a;
        }

        if first_b.is_some() {
            self.b = &self.b[1..];
            return first_b;
        }

        None
    }
}

impl<'a, T> Iterator for VecSymDifference<'a, T>
where
    T: Ord + fmt::Debug,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let mut first_a = self.a.first();
        let mut first_b = self.b.first();

        while first_a == first_b {
            first_a?;
            self.a = &self.a[1..];
            self.b = &self.b[1..];
            first_a = self.a.first();
            first_b = self.b.first();
        }

        if first_a.is_some() && (first_a < first_b || first_b.is_none()) {
            self.a = &self.a[1..];
            return first_a;
        }
        self.b = &self.b[1..];
        first_b
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct IdRangeStep {
    start: u32,
    end: u32,
    step: usize,
}

pub trait IdRange {
    type SelfIter<'a>: Iterator<Item = &'a u32> + Clone
    where
        Self: 'a;
    type DifferenceIter<'a>: Iterator<Item = &'a u32>
    where
        Self: 'a;
    type SymmetricDifferenceIter<'a>: Iterator<Item = &'a u32>
    where
        Self: 'a;
    type IntersectionIter<'a>: Iterator<Item = &'a u32>
    where
        Self: 'a;
    type UnionIter<'a>: Iterator<Item = &'a u32>
    where
        Self: 'a;

    fn new(indexes: Vec<u32>) -> Self;
    fn new_empty() -> Self;
    fn difference<'a>(&'a self, other: &'a Self) -> Self::DifferenceIter<'a>;
    fn symmetric_difference<'a>(&'a self, other: &'a Self) -> Self::SymmetricDifferenceIter<'a>;
    fn intersection<'a>(&'a self, other: &'a Self) -> Self::IntersectionIter<'a>;
    fn union<'a>(&'a self, other: &'a Self) -> Self::UnionIter<'a>;
    fn iter(&self) -> Self::SelfIter<'_>;
    fn contains(&self, id: u32) -> bool;
    fn is_empty(&self) -> bool;
    fn push(&mut self, other: &Self);
    fn push_idrs(&mut self, other: &IdRangeStep);
    fn len(&self) -> usize;
    fn sort(&mut self);
    fn force_sorted(self) -> Self;
}

impl IdRange for IdRangeTree {
    type SelfIter<'a> = btree_set::Iter<'a, u32>;
    type DifferenceIter<'a> = btree_set::Difference<'a, u32>;
    type SymmetricDifferenceIter<'a> = btree_set::SymmetricDifference<'a, u32>;
    type IntersectionIter<'a> = btree_set::Intersection<'a, u32>;
    type UnionIter<'a> = btree_set::Union<'a, u32>;

    fn new(indexes: Vec<u32>) -> IdRangeTree {
        let mut bt = BTreeSet::new();
        bt.extend(&indexes);
        IdRangeTree { indexes: bt }
    }

    fn new_empty() -> Self {
        IdRangeTree {
            indexes: BTreeSet::new(),
        }
    }

    fn difference<'a>(&'a self, other: &'a Self) -> Self::DifferenceIter<'a> {
        self.indexes.difference(&other.indexes)
    }

    fn symmetric_difference<'a>(&'a self, other: &'a Self) -> Self::SymmetricDifferenceIter<'a> {
        self.indexes.symmetric_difference(&other.indexes)
    }

    fn intersection<'a>(&'a self, other: &'a Self) -> Self::IntersectionIter<'a> {
        self.indexes.intersection(&other.indexes)
    }
    fn union<'a>(&'a self, other: &'a Self) -> Self::UnionIter<'a> {
        self.indexes.union(&other.indexes)
    }
    fn iter(&self) -> Self::SelfIter<'_> {
        self.indexes.iter()
    }
    fn contains(&self, id: u32) -> bool {
        self.indexes.contains(&id)
    }
    fn is_empty(&self) -> bool {
        self.indexes.is_empty()
    }
    fn push(&mut self, other: &Self) {
        self.indexes.extend(&other.indexes);
    }
    fn push_idrs(&mut self, idrs: &IdRangeStep) {
        self.indexes
            .extend((idrs.start..idrs.end + 1).step_by(idrs.step))
    }
    fn len(&self) -> usize {
        self.indexes.len()
    }
    fn sort(&mut self) {}
    fn force_sorted(self) -> Self {
        self
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

impl IdRange for IdRangeList {
    type DifferenceIter<'a> = VecDifference<'a, u32>;
    type IntersectionIter<'a> = VecIntersection<'a, u32>;
    type UnionIter<'a> = VecUnion<'a, u32>;
    type SymmetricDifferenceIter<'a> = VecSymDifference<'a, u32>;
    type SelfIter<'a> = std::slice::Iter<'a, u32>;
    fn new(indexes: Vec<u32>) -> IdRangeList {
        IdRangeList {
            indexes,
            sorted: false,
        }
    }
    fn new_empty() -> Self {
        IdRangeList {
            indexes: vec![],
            sorted: true,
        }
    }
    fn push_idrs(&mut self, idrs: &IdRangeStep) {
        self.indexes
            .extend((idrs.start..idrs.end + 1).step_by(idrs.step));
        let sorted = self.sorted && idrs.start > *self.indexes.last().unwrap_or(&0);
        if self.sorted && !sorted {
            self.sorted = false;
            self.sort();
        }
    }

    fn len(&self) -> usize {
        self.indexes.len()
    }

    fn sort(&mut self) {
        if !self.sorted {
            /* println!("sort range"); */
            self.indexes.sort_unstable();
            self.indexes.dedup();
            self.sorted = true;
        }
    }

    fn push(&mut self, other: &Self) {
        /* println!("before: {}", self.sorted); */
        let sorted =
            self.sorted && other.sorted && other.indexes[0] > *self.indexes.last().unwrap_or(&0);
        self.indexes.extend(other.iter());
        if self.sorted && !sorted {
            /* println!("resort"); */
            self.sorted = false;
            self.sort();
        }
        /* println!("after: {}", self.sorted); */
    }

    fn contains(&self, id: u32) -> bool {
        assert!(self.sorted);

        exponential_search(&self.indexes, &id).is_ok()
    }

    fn iter(&self) -> Self::SelfIter<'_> {
        self.indexes.iter()
    }

    fn intersection<'a>(&'a self, other: &'a Self) -> Self::IntersectionIter<'a> {
        assert!(self.sorted);
        assert!(other.sorted);

        Self::IntersectionIter {
            a: &self.indexes,
            b: &other.indexes,
        }
    }

    fn union<'a>(&'a self, other: &'a Self) -> Self::UnionIter<'a> {
        assert!(self.sorted);
        assert!(other.sorted);

        Self::UnionIter {
            a: &self.indexes,
            b: &other.indexes,
        }
    }

    fn symmetric_difference<'a>(&'a self, other: &'a Self) -> Self::SymmetricDifferenceIter<'a> {
        assert!(self.sorted);
        assert!(other.sorted);

        Self::SymmetricDifferenceIter {
            a: &self.indexes,
            b: &other.indexes,
        }
    }

    fn difference<'a>(&'a self, other: &'a Self) -> Self::DifferenceIter<'a> {
        assert!(self.sorted);
        assert!(other.sorted);

        Self::DifferenceIter {
            a: self.indexes.iter(),
            b: other.indexes.iter().peekable(),
        }
    }

    fn is_empty(&self) -> bool {
        self.indexes.is_empty()
    }

    fn force_sorted(mut self) -> Self {
        let s = &mut self;
        s.sorted = true;
        self
    }
}

impl fmt::Display for IdRangeTree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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

impl fmt::Display for IdRangeList {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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

#[derive(Debug, Clone, PartialEq)]
struct IdRangeProduct<T> {
    ranges: Vec<T>,
}

struct IdRangeProductIter<'a, T>
where
    T: IdRange,
{
    ranges: &'a Vec<T>,
    iters: Vec<std::iter::Peekable<T::SelfIter<'a>>>,
}

impl<'a, T> Iterator for IdRangeProductIter<'a, T>
where
    T: IdRange + Clone,
{
    type Item = Vec<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut coords = vec![];
        let mut refill = false;

        if let Some(&coord) = self.iters.last_mut()?.next() {
            coords.push(coord);
        } else {
            *self.iters.last_mut().unwrap() = self.ranges.last()?.iter().peekable();
            coords.push(*self.iters.last_mut()?.next().unwrap());
            refill = true;
        }

        for i in (0..self.ranges.len() - 1).rev() {
            if refill {
                self.iters[i].next();
            }

            if let Some(&&coord) = self.iters[i].peek() {
                coords.push(coord);
                refill = false
            } else {
                self.iters[i] = self.ranges[i].iter().peekable();
                coords.push(**self.iters[i].peek().unwrap());
            }
        }

        if refill {
            None
        } else {
            coords.reverse();
            Some(coords)
        }
    }
}

impl<'a, T> IdRangeProduct<T>
where
    T: IdRange + fmt::Display + Clone + fmt::Debug,
{
    fn intersection(&'a self, other: &'a Self) -> Option<IdRangeProduct<T>> {
        let mut ranges = Vec::<T>::new();

        for (sidr, oidr) in self.ranges.iter().zip(other.ranges.iter()) {
            let rng = T::new(sidr.intersection(oidr).cloned().collect::<Vec<u32>>()).force_sorted();
            if rng.is_empty() {
                return None;
            }
            ranges.push(rng);
        }

        Some(IdRangeProduct { ranges })
    }

    fn difference(&'a self, other: &'a Self) -> Vec<IdRangeProduct<T>> {
        let mut products = vec![];

        let mut next_ranges = self.ranges.clone();
        for (i, (sidr, oidr)) in self.ranges.iter().zip(other.ranges.iter()).enumerate() {
            let rng = T::new(sidr.difference(oidr).cloned().collect::<Vec<u32>>()).force_sorted();
            if rng.is_empty() {
                continue;
            }
            if rng.len() == sidr.len() {
                return vec![self.clone()];
            }

            let mut ranges = next_ranges.clone();
            ranges[i] = rng;
            products.push(IdRangeProduct { ranges });

            next_ranges[i] =
                T::new(oidr.intersection(sidr).cloned().collect::<Vec<u32>>()).force_sorted();
        }

        products
    }

    fn prepare_sort(&mut self) {
        for r in &mut self.ranges {
            r.sort();
        }
    }

    fn iter(&self) -> std::slice::Iter<T> {
        self.ranges.iter()
    }

    fn iter_products(&'a self) -> IdRangeProductIter<'a, T> {
        IdRangeProductIter {
            ranges: &self.ranges,
            iters: self.ranges.iter().map(|r| r.iter().peekable()).collect(),
        }
    }

    fn len(&self) -> usize {
        self.ranges.iter().map(|r| r.len()).product()
    }

    fn num_axis(&self) -> usize {
        self.ranges.len()
    }

    fn fmt_dims(&self, f: &mut fmt::Formatter, dims: &[String]) -> fmt::Result {
        write!(
            f,
            "{}",
            dims.iter()
                .map(|d| d.to_string())
                .interleave(self.ranges.iter().map(|s| s.to_string()))
                .join("")
        )
    }
}

impl<T> fmt::Display for IdRangeProduct<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.ranges.iter().join(""))
    }
}

#[derive(Debug, PartialEq, Clone)]
/// A set of n-dimensional IDs
///
/// Represented internally as a list ok products of ranges of ids
struct IdSet<T> {
    products: Vec<IdRangeProduct<T>>,
}

impl<T> fmt::Display for IdSet<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.products.iter().join(","))
    }
}

struct IdSetIter<'a, T>
where
    T: IdRange + fmt::Display,
{
    product_iter: std::slice::Iter<'a, IdRangeProduct<T>>,
    range_iter: Option<IdRangeProductIter<'a, T>>,
}

impl<'a, T> Iterator for IdSetIter<'a, T>
where
    T: IdRange + fmt::Display + Clone + fmt::Debug,
{
    type Item = Vec<u32>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(coords) = self.range_iter.as_mut()?.next() {
                return Some(coords);
            }
            self.range_iter = Some(self.product_iter.next()?.iter_products());
        }
    }
}

impl<T> IdSet<T>
where
    T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    fn fmt_dims(&self, f: &mut fmt::Formatter, dims: &[String]) -> fmt::Result {
        let mut first = true;
        for p in &self.products {
            if !first {
                write!(f, ",")?;
            }
            p.fmt_dims(f, dims).expect("failed to format string");
            first = false;
        }
        Ok(())
    }

    fn extend(&mut self, other: &Self) {
        self.products.extend(other.products.iter().cloned());
    }

    fn len(&self) -> usize {
        self.products.iter().map(|x| x.len()).sum()
    }

    fn sort(&mut self, skip: usize) {
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
            std::cmp::Ordering::Equal
        });
    }
    fn full_split(&mut self) {
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
                        split_products.push(IdRangeProduct {
                            ranges: vec![T::new(vec![*n]).force_sorted()],
                        });
                    }
                } else {
                    let count = split_products.len() - start;
                    for _ in 1..rng.len() {
                        for j in 0..count {
                            split_products.push(split_products[start + j].clone())
                        }
                    }

                    for (i, r) in rng.iter().enumerate() {
                        for sp in
                            split_products[start + i * count..start + (i + 1) * count].iter_mut()
                        {
                            sp.ranges.push(T::new(vec![*r]).force_sorted());
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

    fn minimal_split(&mut self) {
        /* println!("minimal split"); */
        self.sort(0);
        /* println!("sorted"); */
        let mut idx1 = 0;
        while idx1 < self.products.len() - 1 {
            let mut cur_len = self.products.len();
            let mut idx2 = idx1 + 1;
            while idx2 < cur_len {
                let p1 = &self.products[idx1];
                let p2 = &self.products[idx2];
                /*            println!("compare p1 {} with p2 {}", p1, p2); */
                let mut inter_p = IdRangeProduct::<T> { ranges: vec![] };
                let mut new_products = vec![];
                let mut keep = false;
                let mut term = false;
                let mut intersect = false;
                let mut split = false;

                let num_axis = p1.num_axis();
                for (axis, (r1, r2)) in p1.iter().zip(p2.iter()).enumerate() {
                    intersect = r2.intersection(r1).next().is_some();
                    if !intersect && (axis < num_axis - 1 || !split) {
                        keep = true;
                        if p1.ranges[0].iter().last() < p2.ranges[0].iter().next() {
                            term = true;
                        }
                        break;
                    }

                    if !intersect {
                        //println!("split push");
                        new_products.push(IdRangeProduct {
                            ranges: inter_p.ranges[0..axis]
                                .iter()
                                .chain(&(p1.ranges[axis..]))
                                .cloned()
                                .collect(),
                        });
                        new_products.push(IdRangeProduct {
                            ranges: inter_p.ranges[0..axis]
                                .iter()
                                .chain(&(p2.ranges[axis..]))
                                .cloned()
                                .collect(),
                        });
                    } else if r1 == r2 {
                        //println!("same range {} {}", r1, r2);
                        inter_p.ranges.push(r1.clone());
                    } else {
                        //println!("split range");
                        split = true;
                        inter_p
                            .ranges
                            .push(T::new(r1.intersection(r2).cloned().collect()).force_sorted());
                        if r1.difference(r2).next().is_some() {
                            let diff =
                                vec![T::new(r1.difference(r2).cloned().collect()).force_sorted()];
                            //println!("diff1 {:?}", diff);
                            let new_iter = inter_p.ranges[0..axis]
                                .iter()
                                .chain(&diff)
                                .chain(&(p1.ranges[axis + 1..]));
                            new_products.push(IdRangeProduct {
                                ranges: new_iter.cloned().collect(),
                            });
                        }
                        if r2.difference(r1).next().is_some() {
                            let diff =
                                vec![T::new(r2.difference(r1).cloned().collect()).force_sorted()];
                            //println!("diff2 {:?}", diff);
                            let new_iter = inter_p.ranges[0..axis]
                                .iter()
                                .chain(&diff)
                                .chain(&(p2.ranges[axis + 1..]));
                            new_products.push(IdRangeProduct {
                                ranges: new_iter.cloned().collect(),
                            });
                        }
                    }
                }
                if !keep {
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
                } else {
                    //println!("keep");
                    idx2 += 1;
                }
                if term {
                    //println!("term");
                    break;
                }
            }
            self.sort(idx1);
            //println!("next p1");
            idx1 += 1
        }
    }

    fn merge(&mut self) {
        /* println!("merge"); */
        let mut dellst = vec![false; self.products.len()];

        loop {
            /* println!("loop"); */
            let mut update = false;
            let mut idx1 = 0;
            while idx1 < self.products.len() - 1 {
                if dellst[idx1] {
                    idx1 += 1;
                    continue;
                }
                let mut idx2 = idx1 + 1;
                while idx2 < self.products.len() {
                    if dellst[idx2] {
                        idx2 += 1;
                        continue;
                    }

                    //let mut new_p = vec![];
                    let mut num_diffs = 0;
                    let mut range = 0;
                    {
                        let p2 = &self.products[idx2];
                        let p1 = &self.products[idx1];
                        /*                  println!("try merge p1 {} with p2 {}", p1, p2); */
                        for (i, (r1, r2)) in p1.iter().zip(p2.iter()).enumerate() {
                            if r1 == r2 {
                                /* println!("same range"); */
                            } else if num_diffs == 0 {
                                /* println!("merge range"); */
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
                    }
                    idx2 += 1
                }
                idx1 += 1;
            }
            if !update {
                break;
            }
        }
        self.products = self
            .products
            .iter()
            .cloned()
            .enumerate()
            .filter(|(i, _)| !dellst[*i])
            .map(|(_, p)| p)
            .collect();
    }

    fn fold(&mut self) -> &mut Self {
        for p in &mut self.products {
            p.prepare_sort()
        }

        if self.len() > self.products.len() * self.products.len() {
            /* println!("minimal split"); */
            self.minimal_split();
        } else {
            /* println!("full split"); */
            self.full_split();
        }
        self.merge();

        self
    }

    fn difference(&self, other: &Self) -> Option<Self> {
        let mut products = Vec::<IdRangeProduct<T>>::new();
        for sidpr in self.products.iter() {
            let mut nidpr = vec![sidpr.clone()];
            for oidpr in other.products.iter() {
                nidpr = nidpr.iter().flat_map(|pr| pr.difference(oidpr)).collect();
            }
            products.extend(nidpr)
        }
        if products.is_empty() {
            None
        } else {
            Some(IdSet { products })
        }
    }

    fn intersection(&self, other: &Self) -> Option<Self> {
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
        if products.is_empty() {
            None
        } else {
            Some(IdSet { products })
        }
    }

    fn symmetric_difference(&self, other: &Self) -> Option<Self> {
        let intersection = self.intersection(other);

        let Some(intersection) = intersection else {
            let mut result = self.products.clone();
            result.extend(other.products.iter().cloned());
            return Some(IdSet { products: result })

        };

        let mut ns = self.difference(&intersection);
        let no = other.difference(&intersection);

        match (&mut ns, no) {
            (Some(ns), Some(no)) => {
                ns.extend(&no);
                Some(ns.clone())
            }
            (None, Some(no)) => Some(no),
            (Some(ns), None) => Some(ns.clone()),
            (_, _) => None,
        }
    }

    fn new() -> Self {
        IdSet {
            products: Vec::new(),
        }
    }

    fn iter(&self) -> IdSetIter<'_, T> {
        let mut product_iter = self.products.iter();
        let range_iter = product_iter.next().map(|p| p.iter_products());
        IdSetIter {
            product_iter,
            range_iter,
        }
    }
}

pub struct NodeSetParseError {
    err: String,
}

impl NodeSetParseError {
    fn new(err: String) -> NodeSetParseError {
        NodeSetParseError { err }
    }
}

impl fmt::Debug for NodeSetParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.err)
    }
}

impl fmt::Display for NodeSetParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.err)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeSet<T> {
    dimnames: HashMap<NodeSetDimensions, Option<IdSet<T>>>,
}

struct NodeSetIter<'a, T>
where
    T: IdRange + fmt::Display + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    dim_iter: std::iter::Peekable<
        std::collections::hash_map::Iter<'a, NodeSetDimensions, Option<IdSet<T>>>,
    >,
    set_iter: Option<IdSetIter<'a, T>>,
}

impl<'b, T> NodeSetIter<'b, T>
where
    T: IdRange + fmt::Display + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    fn new(dims: &'b HashMap<NodeSetDimensions, Option<IdSet<T>>>) -> Self {
        let mut it = Self {
            dim_iter: dims.iter().peekable(),
            set_iter: None,
        };
        it.init_dims();
        it
    }

    fn next_dims(&mut self) {
        self.dim_iter.next();
        self.init_dims()
    }

    fn init_dims(&mut self) {
        self.set_iter = self
            .dim_iter
            .peek()
            .and_then(|dims| dims.1.as_ref())
            .map(|s| s.iter());
    }
}

impl<'b, T> Iterator for NodeSetIter<'b, T>
where
    T: IdRange + fmt::Display + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let dimnames = &self.dim_iter.peek()?.0.dimnames;
            let Some(set_iter) = self.set_iter.as_mut() else {
                self.next_dims();
                return Some(dimnames.iter().join(""));
            };

            if let Some(coords) = set_iter.next() {
                /* println!("next coord"); */
                return Some(
                    dimnames
                        .iter()
                        .zip(coords.iter())
                        .map(|(a, b)| format!("{}{}", a, b))
                        .join(""),
                );
            } else {
                /* println!("next dim"); */
                self.next_dims();
            }
        }
    }
}

/// List of names for each dimension of a NodeSet along with an optional suffix
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
struct NodeSetDimensions {
    dimnames: Vec<String>,
    has_suffix: bool,
}

impl NodeSetDimensions {
    /*     fn is_unique(&self) -> bool {
        return self.dimnames.len() == 1 && self.has_suffix
    } */
    fn new() -> NodeSetDimensions {
        NodeSetDimensions {
            dimnames: Vec::<String>::new(),
            has_suffix: false,
        }
    }
    /*todo: better manage has_suffix, check consistency (single suffix)*/
    fn push(&mut self, d: &str, has_suffix: bool) {
        self.dimnames.push(d.into());
        self.has_suffix = has_suffix;
    }
}

impl<T> fmt::Display for NodeSet<T>
where
    T: IdRange + fmt::Display + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut first = true;

        for (dim, set) in &self.dimnames {
            if !first {
                write!(f, ",")?;
            }
            if let Some(set) = set {
                set.fmt_dims(f, &dim.dimnames)
                    .expect("failed to format string");
            } else {
                write!(f, "{}", dim.dimnames[0])?;
            }
            first = false;
        }
        Ok(())
    }
}

impl<T> NodeSet<T>
where
    T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    fn new() -> Self {
        NodeSet {
            dimnames: HashMap::new(),
        }
    }

    fn len(&self) -> usize {
        self.dimnames
            .iter()
            .map(|(_, set)| set.as_ref().map(|s| s.len()).unwrap_or(1))
            .sum()
    }

    fn iter(&self) -> NodeSetIter<'_, T> {
        /* println!("{:?}", self.dimnames); */
        NodeSetIter::new(&self.dimnames)
    }

    fn fold(&mut self) -> &mut Self {
        /* println!("fold {:?}", self.dimnames); */
        self.dimnames.values_mut().for_each(|s| {
            if let Some(s) = s {
                s.fold();
            }
        });

        self
    }

    fn extend(&mut self, other: &Self) {
        for (dimname, oset) in other.dimnames.iter() {
            match self.dimnames.get_mut(dimname) {
                None => {
                    self.dimnames.insert(dimname.clone(), oset.clone());
                }
                Some(set) => {
                    if let Some(s) = set {
                        s.extend(oset.as_ref().unwrap())
                    }
                }
            };
        }
    }

    fn difference(&self, other: &Self) -> Self {
        let mut dimnames = HashMap::<NodeSetDimensions, Option<IdSet<T>>>::new();
        /*    println!("Start intersect"); */
        for (dimname, set) in self.dimnames.iter() {
            /*    println!("{:?}", dimname); */
            if let Some(oset) = other.dimnames.get(dimname) {
                /*    println!("Same dims"); */
                match (set, oset) {
                    (None, None) => continue,
                    (Some(set), Some(oset)) => {
                        if let Some(nset) = set.difference(oset) {
                            /*  println!("Intersect"); */
                            dimnames.insert(dimname.clone(), Some(nset));
                        }
                    }
                    _ => {
                        dimnames.insert(dimname.clone(), set.clone());
                    }
                }
            } else {
                dimnames.insert(dimname.clone(), set.clone());
            }
        }
        NodeSet { dimnames }
    }

    fn intersection(&self, other: &Self) -> Self {
        let mut dimnames = HashMap::<NodeSetDimensions, Option<IdSet<T>>>::new();
        /*    println!("Start intersect"); */
        for (dimname, set) in self.dimnames.iter() {
            /*    println!("{:?}", dimname); */
            if let Some(oset) = other.dimnames.get(dimname) {
                /*    println!("Same dims"); */
                match (set, oset) {
                    (None, None) => {
                        dimnames.insert(dimname.clone(), None);
                    }
                    (Some(set), Some(oset)) => {
                        if let Some(nset) = set.intersection(oset) {
                            /*  println!("Intersect"); */
                            dimnames.insert(dimname.clone(), Some(nset));
                        }
                    }
                    _ => continue,
                }
            }
        }

        NodeSet { dimnames }
    }

    fn symmetric_difference(&self, other: &Self) -> Self {
        let mut dimnames = HashMap::<NodeSetDimensions, Option<IdSet<T>>>::new();
        /*    println!("Start intersect"); */
        for (dimname, set) in self.dimnames.iter() {
            /*    println!("{:?}", dimname); */
            if let Some(oset) = other.dimnames.get(dimname) {
                /*    println!("Same dims"); */
                match (set, oset) {
                    (None, None) => continue,
                    (Some(set), Some(oset)) => {
                        if let Some(nset) = set.symmetric_difference(oset) {
                            /*  println!("Intersect"); */
                            dimnames.insert(dimname.clone(), Some(nset));
                        }
                    }
                    (Some(set), None) => {
                        dimnames.insert(dimname.clone(), Some(set.clone()));
                    }
                    (None, Some(oset)) => {
                        dimnames.insert(dimname.clone(), Some(oset.clone()));
                    }
                }
            }
        }

        NodeSet { dimnames }
    }
}

impl<T> std::str::FromStr for NodeSet<T>
where
    T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    type Err = NodeSetParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parsers::full_expr::<T>(s)
            .map(|r| r.1)
            .map_err(|e| match e {
                nom::Err::Error(e) => NodeSetParseError::new(nom::error::convert_error(s, e)),
                _ => panic!("unreachable"),
            })
    }
}

#[allow(dead_code)]
pub(self) mod parsers {
    use nom::{
        branch::alt,
        bytes::complete::{is_not, tag, take_while1},
        character::complete::{char, digit1, multispace0, multispace1, one_of},
        combinator::{all_consuming, map, map_res, opt, verify},
        error::VerboseError,
        multi::{fold_many0, many0, separated_nonempty_list},
        sequence::{delimited, pair, tuple},
        IResult,
    };
    use std::fmt;
    use std::num::ParseIntError;
    fn not_whitespace(i: &str) -> IResult<&str, &str> {
        is_not(" \t")(i)
    }

    fn is_component_char(c: char) -> bool {
        char::is_alphabetic(c) || ['-', '_', '.'].contains(&c)
    }

    use super::{IdRange, IdRangeProduct, IdRangeStep, IdSet, NodeSet, NodeSetDimensions};

    fn term<T>(i: &str) -> IResult<&str, NodeSet<T>, VerboseError<&str>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        delimited(
            multispace0,
            alt((nodeset, delimited(char('('), expr, char(')')))),
            multispace0,
        )(i)
    }

    pub fn op(i: &str) -> IResult<&str, char, VerboseError<&str>> {
        delimited(multispace0, one_of("+,&!^"), multispace0)(i)
    }
    pub fn space(i: &str) -> IResult<&str, char, VerboseError<&str>> {
        map(multispace1, |_| ' ')(i)
    }

    pub fn full_expr<T>(i: &str) -> IResult<&str, NodeSet<T>, VerboseError<&str>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        all_consuming(expr)(i)
    }

    pub fn expr<T>(i: &str) -> IResult<&str, NodeSet<T>, VerboseError<&str>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        let (i, ns) = term(i)?;
        fold_many0(tuple((opt(op), term)), ns, |mut ns, t| {
            match t.0 {
                Some(',') | Some('+') | None => {
                    ns.extend(&t.1);
                }
                Some('!') => {
                    ns = ns.difference(&t.1);
                }
                Some('^') => {
                    ns = ns.symmetric_difference(&t.1);
                }
                Some('&') => {
                    ns = ns.intersection(&t.1);
                }
                _ => unreachable!(),
            }
            ns
        })(i)
    }

    fn nodeset<T>(i: &str) -> IResult<&str, NodeSet<T>, VerboseError<&str>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        map(
            verify(
                pair(
                    many0(pair(
                        node_component,
                        alt((id_standalone, id_range_bracketed)),
                    )),
                    opt(node_component),
                ),
                |r| !r.0.is_empty() || r.1.is_some(),
            ),
            |r| {
                let mut dims = NodeSetDimensions::new();
                let mut ranges = vec![];
                for (dim, rng) in r.0.into_iter() {
                    let mut range = T::new_empty();
                    for r in rng {
                        range.push_idrs(&r)
                    }
                    ranges.push(range);
                    dims.push(dim, false);
                }
                if let Some(dim) = r.1 {
                    dims.push(dim, true);
                }

                let mut ns = NodeSet::new();
                if ranges.is_empty() {
                    ns.dimnames.entry(dims).or_insert_with(|| None);
                } else {
                    ns.dimnames
                        .entry(dims)
                        .or_insert_with(|| Some(IdSet::new()))
                        .as_mut()
                        .unwrap()
                        .products
                        .push(IdRangeProduct { ranges });
                }
                ns.fold();
                ns
            },
        )(i)
    }

    fn node_component(i: &str) -> IResult<&str, &str, VerboseError<&str>> {
        take_while1(is_component_char)(i)
    }

    fn id_range_bracketed(i: &str) -> IResult<&str, Vec<IdRangeStep>, VerboseError<&str>> {
        delimited(
            char('['),
            separated_nonempty_list(tag(","), id_range_step),
            char(']'),
        )(i)
    }

    fn id_standalone(i: &str) -> IResult<&str, Vec<IdRangeStep>, VerboseError<&str>> {
        map_res(
            digit1,
            |d: &str| -> Result<Vec<IdRangeStep>, ParseIntError> {
                let start = d.parse::<u32>()?;
                Ok(vec![IdRangeStep {
                    start,
                    end: start,
                    step: 1,
                }])
            },
        )(i)
    }

    #[allow(clippy::type_complexity)]
    fn id_range_step(i: &str) -> IResult<&str, IdRangeStep, VerboseError<&str>> {
        map_res(pair(digit1,
                         opt(tuple((
                                 tag("-"),
                                digit1,
                                opt(
                                    pair(
                                        tag("/"),
                                        digit1)))))),

                        |s: (&str, Option<(&str, &str, Option<(&str, &str)>)>) | -> Result<IdRangeStep, ParseIntError>{
                            let start = s.0.parse::<u32>() ?;
                            let (end, step) = match s.1 {
                                None => {(start, 1)},
                                Some(s1) => {
                                    let end = s1.1.parse::<u32>() ?;
                                    match s1.2 {
                                        None => {(end, 1)},
                                        Some((_, step)) => {(end, step.parse::<usize>()?)}
                                    }
                                }
                            };
                            Ok(IdRangeStep{start, end, step})
                        }
                        )(i)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_not_whitespace() {
            assert_eq!(not_whitespace("abcd efg"), Ok((" efg", "abcd")));
            assert_eq!(not_whitespace("abcd\tefg"), Ok(("\tefg", "abcd")));
            assert_eq!(
                not_whitespace(" abcdefg"),
                Err(nom::Err::Error((" abcdefg", nom::error::ErrorKind::IsNot)))
            );
        }

        #[test]
        fn test_id_range_step() {
            assert_eq!(
                id_range_step("2"),
                Ok((
                    "",
                    IdRangeStep {
                        start: 2,
                        end: 2,
                        step: 1
                    }
                ))
            );
            assert_eq!(
                id_range_step("2-34"),
                Ok((
                    "",
                    IdRangeStep {
                        start: 2,
                        end: 34,
                        step: 1
                    }
                ))
            );
            assert_eq!(
                id_range_step("2-34/8"),
                Ok((
                    "",
                    IdRangeStep {
                        start: 2,
                        end: 34,
                        step: 8
                    }
                ))
            );

            assert!(id_range_step("-34/8").is_err());
            assert!(id_range_step("/8").is_err());
            assert_eq!(
                id_range_step("34/8"),
                Ok((
                    "/8",
                    IdRangeStep {
                        start: 34,
                        end: 34,
                        step: 1
                    }
                ))
            );
        }

        #[test]
        fn test_id_range_bracketed() {
            assert_eq!(
                id_range_bracketed("[2]"),
                Ok((
                    "",
                    vec![IdRangeStep {
                        start: 2,
                        end: 2,
                        step: 1
                    }]
                ))
            );
            assert_eq!(
                id_range_bracketed("[2,3-4,5-67/8]"),
                Ok((
                    "",
                    vec![
                        IdRangeStep {
                            start: 2,
                            end: 2,
                            step: 1
                        },
                        IdRangeStep {
                            start: 3,
                            end: 4,
                            step: 1
                        },
                        IdRangeStep {
                            start: 5,
                            end: 67,
                            step: 8
                        }
                    ]
                ))
            );

            assert!(id_range_bracketed("[2,]").is_err());
            assert!(id_range_bracketed("[/8]").is_err());
            assert!(id_range_bracketed("[34-]").is_err());
        }

        #[test]
        fn test_node_component() {
            assert_eq!(node_component("abcd efg"), Ok((" efg", "abcd")));
            assert!(node_component(" abcdefg").is_err());
            assert_eq!(node_component("a_b-c.d2efg"), Ok(("2efg", "a_b-c.d")));
        }
    }
}

use nom::error::VerboseError;

impl From<nom::Err<VerboseError<&str>>> for NodeSetParseError {
    fn from(error: nom::Err<VerboseError<&str>>) -> Self {
        NodeSetParseError::new(format!("{:?}", error))
    }
}

fn main() {
    if let Err(e) = run() {
        println!("Error: {}", e)
    }
}
fn run() -> Result<(), NodeSetParseError> {
    let matches = App::new("ns")
        .subcommand(
            SubCommand::with_name("fold")
                .about("Fold nodeset")
                .arg(
                    Arg::with_name("intersect")
                        .short("i")
                        .multiple(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("nodeset")
                        .required(true)
                        .index(1)
                        .multiple(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("expand")
                .about("Expand nodeset")
                .arg(
                    Arg::with_name("intersect")
                        .short("i")
                        .multiple(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("nodeset")
                        .required(true)
                        .index(1)
                        .multiple(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("count").about("Count nodeset").arg(
                Arg::with_name("nodeset")
                    .required(true)
                    .index(1)
                    .multiple(true),
            ),
        )
        .get_matches();

    if let Some(matches) = matches.subcommand_matches("fold") {
        let nodeset = matches.values_of("nodeset").unwrap().join(" ");
        let mut n: NodeSet<IdRangeList> = nodeset.parse()?;
        n.fold();
        println!("{}", n);
    } else if let Some(matches) = matches.subcommand_matches("expand") {
        let nodeset = matches.values_of("nodeset").unwrap().join(" ");
        let mut n: NodeSet<IdRangeList> = nodeset.parse()?;
        n.fold();
        println!("{}", n.iter().join(" "));
    } else if let Some(matches) = matches.subcommand_matches("count") {
        let nodeset = matches.values_of("nodeset").unwrap().join(" ");
        let nodeset: NodeSet<IdRangeList> = nodeset.parse()?;
        println!("{}", nodeset.len());
    }
    Ok(())
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

    fn prepare_rangesets(count1: u32, count2: u32) -> (IdRangeTree, IdRangeTree) {
        let (v1, v2) = prepare_vectors(count1, count2);
        (IdRangeTree::new(v1.clone()), IdRangeTree::new(v2.clone()))
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
        let (v1, _) = prepare_vectors(DEFAULT_COUNT * 100, DEFAULT_COUNT * 100);
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
            let _rs1 = IdRangeTree::new(v1.clone());
        });
    }

    #[bench]
    fn bench_rangeset_creation_sorted(b: &mut Bencher) {
        let (mut v1, _) = prepare_vectors(DEFAULT_COUNT, DEFAULT_COUNT);
        v1.sort();
        b.iter(|| {
            let _rs1 = IdRangeTree::new(v1.clone());
        });
    }

    #[bench]
    fn bench_rangeset_creation_ranges(b: &mut Bencher) {
        let v1 = prepare_vector_ranges(100, 10);
        b.iter(|| {
            let _rs1 = IdRangeTree::new(v1.clone());
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
        let mut id1: IdSet<IdRangeTree> = IdSet::new();
        let mut id2: IdSet<IdRangeTree> = IdSet::new();

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
            let mut id1: IdSet<IdRangeTree> = IdSet::new();
            id1.push("node[0-100000]");
            id1.full_split();
        });
    }

    #[bench]
    fn bench_idset_merge(b: &mut Bencher) {
        b.iter(|| {
            let mut id1: IdSet<IdRangeTree> = IdSet::new();
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
    fn test_nodeset_parse() {
        let id1: NodeSet<IdRangeList> = "x[1-10/2,5]y[1-7]z3,x[1-10/2,5]y[1-7]z2".parse().unwrap();
        let id2: NodeSet<IdRangeList> = "x[2-5]y[7]z[2,3]".parse().unwrap();

        assert_eq!(
            id1.to_string(),
            "x[1,3,5,7,9]y[1-7]z[3],x[1,3,5,7,9]y[1-7]z[2]"
        );
        assert_eq!(id2.to_string(), "x[2-5]y[7]z[2-3]");
        assert_eq!(
            id1.intersection(&id2).to_string(),
            "x[3,5]y[7]z[3],x[3,5]y[7]z[2]"
        );
    }

    #[test]
    fn test_nodeset_intersect() {
        let id1: NodeSet<IdRangeList> = "x[1-10/2,5]y[1-7]z3,x[1-10/2,5]y[1-7]z2".parse().unwrap();
        let id2: NodeSet<IdRangeList> = "x[2-5]y[7]z[2,3]".parse().unwrap();

        assert_eq!(
            id1.intersection(&id2).fold().to_string(),
            "x[3,5]y[7]z[2-3]"
        );
    }

    #[test]
    fn test_idrangeproduct_iter() {
        /* Empty rangeproduct */
        let mut idpr = IdRangeProduct::<IdRangeList> { ranges: vec![] };

        assert_eq!(
            idpr.iter_products().collect::<Vec<_>>(),
            Vec::<Vec::<u32>>::new()
        );

        /* 1D */
        idpr = IdRangeProduct::<IdRangeList> {
            ranges: vec![IdRangeList::new(vec![0, 1])],
        };

        assert_eq!(
            idpr.iter_products().collect::<Vec<_>>(),
            vec![vec![0], vec![1]]
        );

        /* ND */
        idpr = IdRangeProduct::<IdRangeList> {
            ranges: vec![
                IdRangeList::new(vec![0, 1]),
                IdRangeList::new(vec![2, 3]),
                IdRangeList::new(vec![5, 6]),
            ],
        };

        assert_eq!(
            idpr.iter_products().collect::<Vec<_>>(),
            vec![
                vec![0, 2, 5],
                vec![0, 2, 6],
                vec![0, 3, 5],
                vec![0, 3, 6],
                vec![1, 2, 5],
                vec![1, 2, 6],
                vec![1, 3, 5],
                vec![1, 3, 6]
            ]
        );
    }

    #[test]
    fn test_nodeset_len() {
        let id1: NodeSet<IdRangeList> = "a b".parse().unwrap();
        let id2: NodeSet<IdRangeList> = "a[0-9]b".parse().unwrap();
        let id3: NodeSet<IdRangeList> = "a[0-9]b[0-8]".parse().unwrap();
        let id4: NodeSet<IdRangeList> = "a[0-10000] b[0-100]".parse().unwrap();

        assert_eq!(id1.len(), 2);
        assert_eq!(id2.len(), 10);
        assert_eq!(id3.len(), 90);
        assert_eq!(id4.len(), 10102);
    }

    #[test]
    fn test_nodeset_fold() {
        let mut id1: NodeSet<IdRangeList> =
            "a[1-10/2,5]b[1-7]c3,a[1-10/2,5]b[1-7]c2".parse().unwrap();
        let mut id2: NodeSet<IdRangeList> = "a[0-10]b[0-10],a[0-20]b[0-10]".parse().unwrap();
        let mut id3: NodeSet<IdRangeList> = "x[0-10]y[0-10],x[8-18]y[8-18],x[11-18]y[0-7]"
            .parse()
            .unwrap();

        id1.fold();
        id2.fold();
        id3.fold();

        assert_eq!(id1.to_string(), "a[1,3,5,7,9]b[1-7]c[2-3]");
        assert_eq!(id2.to_string(), "a[0-20]b[0-10]");
        assert_eq!(id3.to_string(), "x[0-7]y[0-10],x[8-18]y[0-18]");
    }

    #[test]
    fn test_nodeset_iter() {
        let id1: NodeSet<IdRangeList> = "a[1-2]b[1-2]".parse().unwrap();

        assert_eq!(
            id1.iter().collect::<Vec<_>>(),
            vec!["a1b1", "a1b2", "a2b1", "a2b2",]
        );
    }

    #[test]
    fn test_exponential_search() {
        assert_eq!(exponential_search(&[], &0), Err(0));
        assert_eq!(exponential_search(&[1, 2, 4, 7], &4), Ok(2));
        assert_eq!(exponential_search(&[1, 2, 4, 7], &0), Err(0));
        assert_eq!(exponential_search(&[1, 2, 4, 7], &8), Err(4));
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
        validate_rangelist_intersection_result(vec![], vec![1, 2, 5, 7], vec![]);
        validate_rangelist_intersection_result(vec![0, 4, 9], vec![], vec![]);
        validate_rangelist_intersection_result(
            vec![0, 4, 9, 7, 12, 34, 35],
            vec![4, 11, 12, 37],
            vec![4, 12],
        );
        validate_rangelist_intersection_result(
            vec![4, 11, 12, 37],
            vec![0, 4, 9, 7, 12, 34, 35],
            vec![4, 12],
        );
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

    #[test]
    #[should_panic]
    fn rangelist_difference_bad1() {
        let rl1 = IdRangeList {
            indexes: vec![1, 2, 3],
            sorted: true,
        };

        let rl2 = IdRangeList {
            indexes: vec![3, 1],
            sorted: false,
        };

        rl1.difference(&rl2);
    }
}
