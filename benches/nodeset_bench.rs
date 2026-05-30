use criterion::{
    black_box, criterion_group, criterion_main, measurement::Measurement, BenchmarkGroup,
    BenchmarkId, Criterion,
};
use nodeset::NodeSet;
use std::time::Duration;

fn set_group_parameters<'a, M>(group: &mut BenchmarkGroup<'a, M>)
where
    M: Measurement,
{
    group.sample_size(50);
    group.measurement_time(Duration::new(5, 0));
    group.warm_up_time(Duration::new(2, 0));
}

fn bench_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse");
    set_group_parameters(&mut group);

    let inputs = [
        ("single", "node1"),
        ("small_range (1K)", "node[1-1000]"),
        ("large_range (1M)", "node[1-1000000]"),
        ("stepped_range (500K over 1M)", "node[1-1000000/2]"),
        ("multi_dim (3d - 48K)", "r[1-100]sw[1-10]-port[1-48]"),
        (
            "multi_dim_stepped_range (4d - 69.7K)",
            "r[1-100/3]sw[1-10/2]-port[1-48/5]-vlan[2-42]",
        ),
        ("comma_list (9)", "node[1,3,5,7,11,13,17,19,23]"),
    ];

    for (name, input) in &inputs {
        group.bench_with_input(BenchmarkId::new("input", name), input, |b, s| {
            b.iter(|| black_box(s.parse::<NodeSet>().unwrap()))
        });
    }

    group.finish();
}

fn bench_display(c: &mut Criterion) {
    let mut group = c.benchmark_group("display");
    set_group_parameters(&mut group);

    let inputs = [
        ("small (1K)", "node[1-1000]"),
        ("large (1M)", "node[1-1000000]"),
        ("stepped_range (500K over 1M)", "node[1-1000000/2]"),
        ("multi_dim (3d - 48K)", "r[1-100]sw[1-10]-port[1-48]"),
        (
            "multi_dim_stepped_range (4d - 69.7K)",
            "r[1-100/3]sw[1-10/2]-port[1-48/5]-vlan[2-42]",
        ),
        ("comma_list (9)", "node[1,3,5,7,11,13,17,19,23]"),
    ];

    for (name, input) in &inputs {
        let ns: NodeSet = input.parse().unwrap();
        group.bench_with_input(BenchmarkId::new("to_string", name), &ns, |b, ns| {
            b.iter(|| black_box(ns.to_string()))
        });
    }

    group.finish();
}

fn bench_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("operations");
    set_group_parameters(&mut group);

    // Overlapping ranges: meaningful intersection
    let ns_a: NodeSet = "node[1-10000]".parse().unwrap();
    let ns_b: NodeSet = "node[5000-15000]".parse().unwrap();

    group.bench_function("union/10k+10k", |b| b.iter(|| black_box(ns_a.union(&ns_b))));

    group.bench_function("intersection/10k+10k", |b| {
        b.iter(|| black_box(ns_a.intersection(&ns_b)))
    });

    group.bench_function("difference/10k-10k", |b| {
        b.iter(|| black_box(ns_a.difference(&ns_b)))
    });

    // Disjoint sets: empty intersection
    let ns_c: NodeSet = "node[1-10000]".parse().unwrap();
    let ns_d: NodeSet = "node[20000-30000]".parse().unwrap();

    group.bench_function("union/disjoint", |b| {
        b.iter(|| black_box(ns_c.union(&ns_d)))
    });

    group.bench_function("intersection/disjoint", |b| {
        b.iter(|| black_box(ns_c.intersection(&ns_d)))
    });

    group.finish();
}

fn bench_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("iter");
    set_group_parameters(&mut group);

    let inputs = [
        ("small (1K)", "node[1-1000]"),
        ("large (1M)", "node[1-1000000]"),
        ("stepped_range (500K over 1M)", "node[1-1000000/2]"),
        ("multi_dim (3d - 48K)", "r[1-100]sw[1-10]-port[1-48]"),
        (
            "multi_dim_stepped (4d - 69.7K)",
            "r[1-100/3]sw[1-10/2]-port[1-48/5]-vlan[2-42]",
        ),
        ("comma_list (9)", "node[1,3,5,7,11,13,17,19,23]"),
    ];

    for (name, input) in &inputs {
        let ns: NodeSet = input.parse().unwrap();
        group.bench_with_input(BenchmarkId::new("collect", name), &ns, |b, ns| {
            b.iter(|| black_box(ns.iter().collect::<Vec<_>>()))
        });
    }

    group.finish();
}

fn bench_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("count");
    set_group_parameters(&mut group);

    let ns_large: NodeSet = "node[1-1000000]".parse().unwrap();
    let ns_multi: NodeSet = "r[1-100]sw[1-10]-port[1-48]".parse().unwrap();
    let ns_multi_stepped: NodeSet = "r[1-100/3]sw[1-10/2]-port[1-48/5]-vlan[2-42]"
        .parse()
        .unwrap();

    group.bench_function("count/large (1M)", |b| b.iter(|| black_box(ns_large.len())));

    group.bench_function("count/multi_dim (3d - 48K)", |b| {
        b.iter(|| black_box(ns_multi.len()))
    });

    group.bench_function("count/multi_dim_stepped (4d - 69.7K)", |b| {
        b.iter(|| black_box(ns_multi_stepped.len()))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_parse,
    bench_display,
    bench_operations,
    bench_iter,
    bench_count,
);
criterion_main!(benches);
