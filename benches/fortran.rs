use test::Bencher;

use blas::fortran::dgemv;

#[bench]
fn dgemv_00010(bencher: &mut Bencher) {
    run(10, bencher)
}

#[bench]
fn dgemv_00100(bencher: &mut Bencher) {
    run(100, bencher)
}

#[bench]
fn dgemv_01000(bencher: &mut Bencher) {
    run(1000, bencher)
}

#[bench]
fn dgemv_10000(bencher: &mut Bencher) {
    run(10000, bencher)
}

fn run(m: i32, bencher: &mut Bencher) {
    let a = vec![1.0; (m * m) as usize];
    let x = vec![1.0; m as usize];
    let mut y = vec![1.0; m as usize];

    bencher.iter(|| unsafe {
        dgemv(b'N', m, m, 1.0, &a, m, &x, 1, 1.0, &mut y, 1)
    });
}
