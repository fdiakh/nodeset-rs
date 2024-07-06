use crate::idrange::IdRange;
use itertools::Itertools;
use log::trace;
use std::fmt::{self, Debug, Display};

use super::nodeset::NodeSetDimensions;

/// A product of IdRanges over multiple dimensions
#[derive(Debug, Clone, Eq, PartialEq, Default)]
pub struct IdRangeProduct<T> {
    pub(crate) ranges: Vec<T>,
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
        let mut coords = vec![0; self.ranges.len()];
        let mut refill = false;

        if let Some(&coord) = self.iters.last_mut()?.next() {
            coords[self.ranges.len() - 1] = coord;
        } else {
            *self.iters.last_mut().unwrap() = self.ranges.last()?.iter().peekable();
            coords[self.ranges.len() - 1] = *self.iters.last_mut()?.next().unwrap();
            refill = true;
        }

        for i in (0..self.ranges.len() - 1).rev() {
            if refill {
                self.iters[i].next();
            }

            if let Some(&&coord) = self.iters[i].peek() {
                coords[i] = coord;
                refill = false
            } else {
                self.iters[i] = self.ranges[i].iter().peekable();
                coords[i] = **self.iters[i].peek().unwrap();
            }
        }

        if refill {
            None
        } else {
            Some(coords)
        }
    }
}

impl<'a, T> IdRangeProduct<T>
where
    T: IdRange + Display + Clone + Debug,
{
    fn intersection(&'a self, other: &'a Self) -> Option<IdRangeProduct<T>> {
        let mut ranges = Vec::<T>::new();

        for (sidr, oidr) in self.ranges.iter().zip(other.ranges.iter()) {
            let rng = T::from_sorted(sidr.intersection(oidr));
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
            let rng = T::from_sorted(sidr.difference(oidr));
            if rng.is_empty() {
                continue;
            }
            if rng.len() == sidr.len() {
                return vec![self.clone()];
            }

            let mut ranges = next_ranges.clone();
            ranges[i] = rng;
            products.push(IdRangeProduct { ranges });

            next_ranges[i] = T::from_sorted(oidr.intersection(sidr));
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
}

impl<T> Display for IdRangeProduct<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.ranges.iter().join(""))
    }
}

#[derive(Debug, Eq, PartialEq, Clone, Default)]
/// A set of n-dimensional IDs
///
/// Represented internally as a list of products of ranges of ids
pub struct IdSet<T> {
    pub(crate) products: Vec<IdRangeProduct<T>>,
}

impl<T> Display for IdSet<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.products.iter().join(","))
    }
}

pub struct IdSetIter<'a, T>
where
    T: IdRange + Display,
{
    product_iter: std::slice::Iter<'a, IdRangeProduct<T>>,
    range_iter: Option<IdRangeProductIter<'a, T>>,
}

impl<'a, T> Iterator for IdSetIter<'a, T>
where
    T: IdRange + Display + Clone + Debug,
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
    T: IdRange + PartialEq + Clone + Display + Debug,
{
    pub(crate) fn fmt_dims(&self, f: &mut fmt::Formatter, dims: &NodeSetDimensions) -> fmt::Result {
        let mut first = true;
        for p in &self.products {
            if !first {
                f.write_str(",")?;
            }
            dims.fmt_ranges(f, &p.ranges)?;

            first = false;
        }
        Ok(())
    }

    pub fn extend(&mut self, other: &Self) {
        self.products.extend(other.products.iter().cloned());
    }

    pub fn len(&self) -> usize {
        self.products.iter().map(|x| x.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.products.is_empty()
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

    /// Split all products into products with one element each
    fn full_split(&mut self) {
        trace!("Splitting products into single elements");

        let mut split_products = vec![];
        for p in &self.products {
            let mut first_range = true;
            let start = split_products.len();
            for rng in &p.ranges {
                if first_range {
                    first_range = false;
                    for n in rng.iter() {
                        split_products.push(IdRangeProduct {
                            ranges: vec![T::from(*n)],
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
                            sp.ranges.push(T::from(*r));
                        }
                    }
                }
            }
        }
        self.products = split_products;

        self.sort(0);
        self.products.dedup();
    }

    /// Split products into mergeable products by creating products with common
    /// ranges.
    ///
    /// Prioritize splitting ranges along the first dimensions to be able
    /// to the create largest ranges possible along the last dimensions in the
    /// merging step.
    fn minimal_split(&mut self) {
        trace!("Splitting products to create common ranges");

        self.sort(0);
        let mut idx1 = 0;
        while idx1 + 1 < self.products.len() {
            let mut cur_len = self.products.len();
            let mut idx2 = idx1 + 1;
            while idx2 < cur_len {
                let p1 = &self.products[idx1];
                let p2 = &self.products[idx2];

                let mut inter_p = IdRangeProduct::<T> { ranges: vec![] };
                let mut new_products = vec![];
                let mut keep = false;
                let mut term = false;
                let mut intersect = false;
                let mut split = false;

                trace!("Checking if we can split p1: {p1} and p2: {p2} to create a common range",);
                let num_axis = p1.num_axis();
                for (axis, (r1, r2)) in p1.iter().zip(p2.iter()).enumerate() {
                    intersect = r2.intersection(r1).next().is_some();
                    if !intersect && (axis < num_axis - 1 || !split) {
                        trace!("Dimension {axis} does not intersect and we have not split yet or are finished");
                        keep = true;
                        if p1.ranges[0].iter().last() < p2.ranges[0].iter().next() {
                            trace!("Further products cannot intersect with p1");
                            term = true;
                        }
                        break;
                    }

                    if !intersect {
                        trace!("Dimension {axis} does not intersect but we have a previous split");
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
                        trace!(
                            "Pushed new products {} {}",
                            new_products[new_products.len() - 2],
                            new_products[new_products.len() - 1]
                        );
                    } else if r1 == r2 {
                        trace!("Dimension {axis} is common");
                        inter_p.ranges.push(r1.clone());
                    } else {
                        trace!("Dimension {axis} intersects: split products");
                        split = true;
                        inter_p.ranges.push(T::from_sorted(r1.intersection(r2)));
                        if r1.difference(r2).next().is_some() {
                            let diff = vec![T::from_sorted(r1.difference(r2))];
                            let new_iter = inter_p.ranges[0..axis]
                                .iter()
                                .chain(&diff)
                                .chain(&(p1.ranges[axis + 1..]));
                            new_products.push(IdRangeProduct {
                                ranges: new_iter.cloned().collect(),
                            });
                            trace!(
                                "Pushed new product {}",
                                new_products[new_products.len() - 1]
                            );
                        }
                        if r2.difference(r1).next().is_some() {
                            let diff = vec![T::from_sorted(r2.difference(r1))];
                            let new_iter = inter_p.ranges[0..axis]
                                .iter()
                                .chain(&diff)
                                .chain(&(p2.ranges[axis + 1..]));
                            new_products.push(IdRangeProduct {
                                ranges: new_iter.cloned().collect(),
                            });
                            trace!(
                                "Pushed new product {}",
                                new_products[new_products.len() - 1]
                            );
                        }

                        trace!("In progress intersection {}", inter_p);
                    }
                }
                if !keep {
                    trace!("Deleting p1 and p2");
                    if intersect {
                        self.products[idx1] = inter_p;
                    } else {
                        self.products[idx1] = new_products[0].clone();
                        new_products.swap_remove(0);
                    }
                    if idx2 < cur_len - 1 {
                        self.products.swap(idx2, cur_len - 1);
                    }
                    self.products.swap_remove(cur_len - 1);
                    cur_len -= 1;
                    self.products.append(&mut new_products);
                } else {
                    trace!("Keeeping original p1 and p2");
                    idx2 += 1;
                }
                if term {
                    trace!("Skipping comparison with other products which cannot intersect");
                    break;
                }
            }
            self.sort(idx1);
            trace!("Next products");
            idx1 += 1
        }
    }

    /// Merge products with common ranges.
    fn merge(&mut self) {
        let mut dellst = vec![false; self.products.len()];
        let mut merge_range = self.products[0].num_axis() - 1;

        // Loop until we don't find anything to merge
        // Prioritize merging axis per axis from the last to the first
        loop {
            let mut update = false;
            let mut idx1 = 0;
            while idx1 + 1 < self.products.len() {
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

                    let mut num_diffs = 0;
                    let mut range = 0;
                    {
                        let p2 = &self.products[idx2];
                        let p1 = &self.products[idx1];
                        trace!("try merge p1: {p1} with p2: {p2} on range {merge_range}");
                        for (i, (r1, r2)) in p1.iter().zip(p2.iter()).enumerate() {
                            if r1 == r2 {
                                trace!("Range {i} is common");
                            } else if num_diffs == 0 {
                                trace!("Range {i} is the first to differ");
                                num_diffs += 1;
                                range = i;
                                if range != merge_range {
                                    trace!("Not the range we want at this stage: abort");
                                    break;
                                }
                            } else {
                                trace!("More than one difference: abort");
                                num_diffs += 1;
                                break;
                            }
                        }
                    }
                    if num_diffs < 2 && range == merge_range {
                        trace!("Merge both products with only differ in range {range}");
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
                if merge_range == 0 {
                    break;
                } else {
                    merge_range -= 1;
                }
            } else {
                merge_range = self.products[0].num_axis() - 1;
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

    pub fn fold(&mut self) -> &mut Self {
        for p in &mut self.products {
            p.prepare_sort()
        }

        // This is a heuristic to determine whether to do a full split or a
        // minimal split. The minimal split algorithm is O(n^2) in the number of
        // products, so it's not worth using it if the total number of elements
        // is smaller than the number of products squared. In that case, we do a
        // full split, which is O(n) in the number of elements
        if self.len() > self.products.len() * self.products.len() {
            self.minimal_split();
        } else {
            self.full_split();
        }
        self.merge();

        self
    }

    pub fn difference(&self, other: &Self) -> Option<Self> {
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

    pub fn intersection(&self, other: &Self) -> Option<Self> {
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

    pub fn symmetric_difference(&self, other: &Self) -> Option<Self> {
        let intersection = self.intersection(other);

        let Some(intersection) = intersection else {
            let mut result = self.products.clone();
            result.extend(other.products.iter().cloned());
            return Some(IdSet { products: result });
        };

        let ns = self.difference(&intersection);
        let no = other.difference(&intersection);

        match (ns, no) {
            (Some(mut ns), Some(no)) => {
                ns.extend(&no);
                ns.fold();
                Some(ns)
            }
            (None, Some(no)) => Some(no),
            (Some(ns), None) => Some(ns),
            (_, _) => None,
        }
    }

    pub fn new() -> Self {
        IdSet {
            products: Vec::new(),
        }
    }

    pub fn iter(&self) -> IdSetIter<'_, T> {
        let mut product_iter = self.products.iter();
        let range_iter = product_iter.next().map(|p| p.iter_products());
        IdSetIter {
            product_iter,
            range_iter,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idrange::IdRangeList;
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
            ranges: vec![IdRangeList::from(vec![0, 1])],
        };

        assert_eq!(
            idpr.iter_products().collect::<Vec<_>>(),
            vec![vec![0], vec![1]]
        );

        /* ND */
        idpr = IdRangeProduct::<IdRangeList> {
            ranges: vec![
                IdRangeList::from(vec![0, 1]),
                IdRangeList::from(vec![2, 3]),
                IdRangeList::from(vec![5, 6]),
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
}
