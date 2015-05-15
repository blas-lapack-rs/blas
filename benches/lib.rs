#![feature(test)]

extern crate blas;
extern crate test;

use std::iter::repeat;

#[bench]
fn dgemv_few_large(bench: &mut test::Bencher) {
    let m = 1000;

    let a = repeat(1.0).take(m * m).collect::<Vec<_>>();
    let x = repeat(1.0).take(m * 1).collect::<Vec<_>>();
    let mut y = repeat(1.0).take(m * 1).collect::<Vec<_>>();

    bench.iter(|| {
        blas::dgemv(blas::Trans::N, m, m, 1.0, &a, m, &x, 1, 1.0, &mut y, 1)
    });
}

#[bench]
fn dgemv_many_small(bench: &mut test::Bencher) {
    let m = 20;

    let a = repeat(1.0).take(m * m).collect::<Vec<_>>();
    let x = repeat(1.0).take(m * 1).collect::<Vec<_>>();
    let mut y = repeat(1.0).take(m * 1).collect::<Vec<_>>();

    bench.iter(|| {
        for _ in 0..20000 {
            blas::dgemv(blas::Trans::N, m, m, 1.0, &a, m, &x, 1, 1.0, &mut y, 1);
        }
    });
}
