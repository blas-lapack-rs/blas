//! An interface to the [Basic Linear Algebra Subprograms][1].
//!
//! [1]: http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms

#![cfg_attr(test, feature(test))]

#[cfg(test)]
#[macro_use]
extern crate assert;

#[cfg(test)]
extern crate test;

extern crate "libblas-sys" as raw;

pub enum Layout {
    RowMajor = raw::CblasRowMajor as isize,
    ColumnMajor = raw::CblasColMajor as isize,
}

pub enum Transpose {
    None = raw::CblasNoTrans as isize,
    Transpose = raw::CblasTrans as isize,
    ConjugateTranspose = raw::CblasConjTrans as isize,
}

#[inline]
pub fn dgemv(layout: Layout, trans: Transpose, m: usize, n: usize, alpha: f64,
             a: &[f64], lda: usize, x: &[f64], incx: usize, beta: f64,
             y: &mut [f64], incy: usize) {

    unsafe {
        raw::cblas_dgemv(layout as u32, trans as u32, m as i32, n as i32, alpha,
                         a.as_ptr(), lda as i32, x.as_ptr(), incx as i32, beta,
                         y.as_mut_ptr(), incy as i32);
    }
}

#[inline]
pub fn dgemm(layout: Layout, transa: Transpose, transb: Transpose, m: usize,
             n: usize, k: usize, alpha: f64, a: &[f64], lda: usize, b: &[f64],
             ldb: usize, beta: f64, c: &mut [f64], ldc: usize) {

    unsafe {
        raw::cblas_dgemm(layout as u32, transa as u32, transb as u32,
                         m as i32, n as i32, k as i32, alpha, a.as_ptr(),
                         lda as i32, b.as_ptr(), ldb as i32, beta,
                         c.as_mut_ptr(), ldc as i32);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn dgemv() {
        let (m, n) = (2, 3);

        let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![6.0, 8.0];

        ::dgemv(::Layout::ColumnMajor, ::Transpose::None,
                m, n, 1.0, &a, m, &x, 1, 1.0, &mut y, 1);

        let expected_y = vec![20.0, 40.0];
        assert_equal!(y, expected_y);
    }

    #[test]
    fn dgemm() {
        let (m, n, k) = (2, 4, 3);

        let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let b = vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0];
        let mut c = vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0];

        ::dgemm(::Layout::ColumnMajor, ::Transpose::None, ::Transpose::None,
                m, n, k, 1.0, &a, m, &b, k, 1.0, &mut c, m);

        let expected_c = vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0];
        assert_equal!(c, expected_c);
    }
}

#[cfg(test)]
mod benches {
    use std::iter::repeat;

    #[bench]
    fn dgemv_few_large(bench: &mut ::test::Bencher) {
        let m = 1000;

        let a = repeat(1.0).take(m * m).collect::<Vec<_>>();
        let x = repeat(1.0).take(m).collect::<Vec<_>>();
        let mut y = repeat(1.0).take(m).collect::<Vec<_>>();

        bench.iter(|| {
            ::dgemv(::Layout::ColumnMajor, ::Transpose::None,
                    m, m, 1.0, &a, m, &x, 1, 1.0, &mut y, 1)
        });
    }

    #[bench]
    fn dgemv_many_small(bench: &mut ::test::Bencher) {
        let m = 20;

        let a = repeat(1.0).take(m * m).collect::<Vec<_>>();
        let x = repeat(1.0).take(m).collect::<Vec<_>>();
        let mut y = repeat(1.0).take(m).collect::<Vec<_>>();

        bench.iter(|| {
            for _ in 0..20000 {
                ::dgemv(::Layout::ColumnMajor, ::Transpose::None,
                        m, m, 1.0, &a, m, &x, 1, 1.0, &mut y, 1);
            }
        });
    }
}
