extern crate blas;
extern crate test;

use test::Bencher;

#[bench]
fn dgemv_few_large(b: &mut Bencher) {
    let m = 1000;

    let a = Vec::from_elem(m * m, 1.0);
    let x = Vec::from_elem(m * 1, 1.0);
    let mut y = Vec::from_elem(m * 1, 1.0);

    b.iter(|| {
        blas::dgemv('N' as i8, m as i32, m as i32, 1.0, a.as_ptr(),
            m as i32, x.as_ptr(), 1, 1.0, y.as_mut_ptr(), 1)
    })
}

#[bench]
fn dgemv_many_small(b: &mut Bencher) {
    let m = 20;

    let a = Vec::from_elem(m * m, 1.0);
    let x = Vec::from_elem(m * 1, 1.0);
    let mut y = Vec::from_elem(m * 1, 1.0);

    b.iter(|| {
        for _ in range(0u, 20000) {
            blas::dgemv('N' as i8, m as i32, m as i32, 1.0, a.as_ptr(),
                m as i32, x.as_ptr(), 1, 1.0, y.as_mut_ptr(), 1);
        }
    })
}
