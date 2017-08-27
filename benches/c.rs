use test::Bencher;

use blas::c::{dgemv, Layout, Transpose};

#[bench]
fn dgemv_cm_00010(bencher: &mut Bencher) {
    run(Layout::ColumnMajor, 10, bencher)
}

#[bench]
fn dgemv_cm_00100(bencher: &mut Bencher) {
    run(Layout::ColumnMajor, 100, bencher)
}

#[bench]
fn dgemv_cm_01000(bencher: &mut Bencher) {
    run(Layout::ColumnMajor, 1000, bencher)
}

#[bench]
fn dgemv_cm_10000(bencher: &mut Bencher) {
    run(Layout::ColumnMajor, 10000, bencher)
}

#[bench]
fn dgemv_rm_00010(bencher: &mut Bencher) {
    run(Layout::RowMajor, 10, bencher)
}

#[bench]
fn dgemv_rm_00100(bencher: &mut Bencher) {
    run(Layout::RowMajor, 100, bencher)
}

#[bench]
fn dgemv_rm_01000(bencher: &mut Bencher) {
    run(Layout::RowMajor, 1000, bencher)
}

#[bench]
fn dgemv_rm_10000(bencher: &mut Bencher) {
    run(Layout::RowMajor, 10000, bencher)
}

fn run(layout: Layout, m: i32, bencher: &mut Bencher) {
    let a = vec![1.0; (m * m) as usize];
    let x = vec![1.0; m as usize];
    let mut y = vec![1.0; m as usize];

    bencher.iter(|| unsafe {
        dgemv(
            layout,
            Transpose::None,
            m,
            m,
            1.0,
            &a,
            m,
            &x,
            1,
            1.0,
            &mut y,
            1,
        )
    });
}
