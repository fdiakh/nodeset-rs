use crate::idrange::IdRange;
use itertools::Itertools;
use std::fmt::{self, Debug, Display};

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
    T: IdRange + Display + Clone + Debug,
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

impl<T> Display for IdRangeProduct<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.ranges.iter().join(""))
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
        write!(f, "{}", self.products.iter().join(","))
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
    pub fn fmt_dims(&self, f: &mut fmt::Formatter, dims: &[String]) -> fmt::Result {
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
    fn full_split(&mut self) {
        /*  println!("full split"); */
        let mut split_products = vec![];
        for p in &self.products {
            // Each product is split into a list of products with only on element each
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
        while idx1 + 1 < self.products.len() {
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

    pub fn fold(&mut self) -> &mut Self {
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
}
