#![feature(test)]

extern crate blas;
extern crate test;

use test::Bencher;

#[bench]
fn dgemv_few_large(bencher: &mut Bencher) {
    let m = 1000;

    let a = vec![1.0; m * m];
    let x = vec![1.0; m];
    let mut y = vec![1.0; m];

    bencher.iter(|| {
        blas::dgemv(b'N', m, m, 1.0, &a, m, &x, 1, 1.0, &mut y, 1)
    });
}

#[bench]
fn dgemv_many_small(bencher: &mut Bencher) {
    let m = 20;

    let a = vec![1.0; m * m];
    let x = vec![1.0; m];
    let mut y = vec![1.0; m];

    bencher.iter(|| {
        for _ in 0..20000 {
            blas::dgemv(b'N', m, m, 1.0, &a, m, &x, 1, 1.0, &mut y, 1);
        }
    });
}
