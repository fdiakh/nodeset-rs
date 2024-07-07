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
    coords: Vec<u32>,
    iters: Vec<std::iter::Peekable<T::SelfIter<'a>>>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ProductCoords {
    Pair([u32; 2]),
    Triple([u32; 3]),
    Dynamic(Vec<u32>),
}

impl ProductCoords {
    pub(crate) fn iter(&self) -> ProductCoordsIter {
        ProductCoordsIter {
            coords: self,
            idx: 0,
        }
    }
}

#[derive(Debug, PartialEq)]
pub(crate) struct ProductCoordsIter<'a> {
    coords: &'a ProductCoords,
    idx: usize,
}

impl Iterator for ProductCoordsIter<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.idx;
        self.idx += 1;
        match self.coords {
            ProductCoords::Pair(ref c) => c.get(idx).copied(),
            ProductCoords::Triple(ref c) => c.get(idx).copied(),
            ProductCoords::Dynamic(ref c) => c.get(idx).copied(),
        }
    }
}

impl<'a, T> Iterator for IdRangeProductIter<'a, T>
where
    T: IdRange + Clone,
{
    type Item = ProductCoords;

    fn next(&mut self) -> Option<Self::Item> {
        let coords = &mut self.coords;
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
        } else if self.ranges.len() == 2 {
            Some(ProductCoords::Pair([coords[0], coords[1]]))
        } else if self.ranges.len() == 3 {
            Some(ProductCoords::Triple([coords[0], coords[1], coords[2]]))
        } else {
            Some(ProductCoords::Dynamic(coords.clone()))
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
            coords: vec![0; self.ranges.len()],
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
pub(crate) struct IdSet<T> {
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

pub(crate) struct IdSetIter<'a, T>
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
    type Item = ProductCoords;
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

        let mut orig_products = vec![];

        std::mem::swap(&mut orig_products, &mut self.products);

        for p in orig_products.into_iter() {
            if p.len() == 1 {
                self.products.push(p);
                continue;
            }

            let mut first_range = true;
            let start = self.products.len();
            for rng in p.ranges {
                if first_range {
                    first_range = false;
                    for n in rng.iter() {
                        self.products.push(IdRangeProduct {
                            ranges: vec![T::from(*n)],
                        });
                    }
                } else {
                    let count = self.products.len() - start;
                    for _ in 1..rng.len() {
                        for j in 0..count {
                            self.products.push(self.products[start + j].clone())
                        }
                    }

                    for (i, r) in rng.iter().enumerate() {
                        for sp in
                            self.products[start + i * count..start + (i + 1) * count].iter_mut()
                        {
                            sp.ranges.push(T::from(*r));
                        }
                    }
                }
            }
        }

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
        let mut merge_axis = self.products[0].num_axis();
        let mut merged = 0;
        // Prioritize merging axis per axis from the last to the first
        while merge_axis > 0 {
            merge_axis -= 1;
            let mut idx1 = 0;
            while idx1 + 1 < self.products.len() {
                let mut done = false;
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

                    let mut mergeable = true;
                    {
                        let p2 = &self.products[idx2];
                        let p1 = &self.products[idx1];
                        trace!("try merge p1: {p1} with p2: {p2} on axis {merge_axis}");
                        for (i, (r1, r2)) in p1.iter().zip(p2.iter()).enumerate() {
                            if i != merge_axis && r1 != r2 {
                                trace!("Axis {i} differs but we need to merge on axis {merge_axis}: abort");
                                mergeable = false;

                                if i < merge_axis && r2.iter().next() > r1.iter().next() {
                                    trace!("Other products cannot intersect with us");
                                    done = true;
                                }
                                break;
                            }
                        }
                    }
                    if mergeable {
                        trace!("Merge both products which only differ in axis {merge_axis}");
                        let (pp1, pp2) = self.products.split_at_mut(idx2);
                        pp1[idx1].ranges[merge_axis].set_lazy();
                        pp1[idx1].ranges[merge_axis].push(&pp2[0].ranges[merge_axis]);
                        dellst[idx2] = true;
                        merged += 1;
                    }
                    if done {
                        break;
                    }
                    idx2 += 1
                }
                self.products[idx1].ranges[merge_axis].sort();
                idx1 += 1;
            }
        }

        let mut orig_products = Vec::with_capacity(self.products.len() - merged);

        std::mem::swap(&mut self.products, &mut orig_products);

        self.products.extend(
            orig_products
                .into_iter()
                .enumerate()
                .filter(|(i, _)| !dellst[*i])
                .map(|(_, p)| p),
        );
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
            Vec::<ProductCoords>::new()
        );

        /* 2D */
        idpr = IdRangeProduct::<IdRangeList> {
            ranges: vec![IdRangeList::from(vec![0, 1]), IdRangeList::from(vec![2, 3])],
        };

        assert_eq!(
            idpr.iter_products().collect::<Vec<_>>(),
            vec![
                ProductCoords::Pair([0, 2]),
                ProductCoords::Pair([0, 3]),
                ProductCoords::Pair([1, 2]),
                ProductCoords::Pair([1, 3])
            ]
        );

        /* 3D */
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
                ProductCoords::Triple([0, 2, 5]),
                ProductCoords::Triple([0, 2, 6]),
                ProductCoords::Triple([0, 3, 5]),
                ProductCoords::Triple([0, 3, 6]),
                ProductCoords::Triple([1, 2, 5]),
                ProductCoords::Triple([1, 2, 6]),
                ProductCoords::Triple([1, 3, 5]),
                ProductCoords::Triple([1, 3, 6])
            ]
        );

        /* ND */
        idpr = IdRangeProduct::<IdRangeList> {
            ranges: vec![
                IdRangeList::from(vec![0, 1]),
                IdRangeList::from(vec![2, 3]),
                IdRangeList::from(vec![5, 6]),
                IdRangeList::from(vec![7, 8]),
            ],
        };

        assert_eq!(
            idpr.iter_products().collect::<Vec<_>>(),
            vec![
                ProductCoords::Dynamic(vec![0, 2, 5, 7]),
                ProductCoords::Dynamic(vec![0, 2, 5, 8]),
                ProductCoords::Dynamic(vec![0, 2, 6, 7]),
                ProductCoords::Dynamic(vec![0, 2, 6, 8]),
                ProductCoords::Dynamic(vec![0, 3, 5, 7]),
                ProductCoords::Dynamic(vec![0, 3, 5, 8]),
                ProductCoords::Dynamic(vec![0, 3, 6, 7]),
                ProductCoords::Dynamic(vec![0, 3, 6, 8]),
                ProductCoords::Dynamic(vec![1, 2, 5, 7]),
                ProductCoords::Dynamic(vec![1, 2, 5, 8]),
                ProductCoords::Dynamic(vec![1, 2, 6, 7]),
                ProductCoords::Dynamic(vec![1, 2, 6, 8]),
                ProductCoords::Dynamic(vec![1, 3, 5, 7]),
                ProductCoords::Dynamic(vec![1, 3, 5, 8]),
                ProductCoords::Dynamic(vec![1, 3, 6, 7]),
                ProductCoords::Dynamic(vec![1, 3, 6, 8])
            ]
        );
    }
}
