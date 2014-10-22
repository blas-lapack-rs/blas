//! An interface to the [Basic Linear Algebra Subprograms][1].
//!
//! [1]: http://www.netlib.org/blas/

#![feature(macro_rules)]

extern crate libc;

use libc::{c_char, c_double, c_int};

#[link(name = "gfortran")]
#[link(name = "blas", kind = "static")]
extern {
    fn dgemv_(trans: *const c_char, m: *const c_int, n: *const c_int, alpha: *const c_double,
              a: *const c_double, lda: *const c_int, x: *const c_double, incx: *const c_int,
              beta: *const c_double, y: *mut c_double, incy: *const c_int);

    fn dgemm_(transa: *const c_char, transb: *const c_char, m: *const c_int, n: *const c_int,
              k: *const c_int, alpha: *const c_double, a: *const c_double, lda: *const c_int,
              b: *const c_double, ldb: *const c_int, beta: *const c_double, c: *mut c_double,
              ldc: *const c_int);
}

/// Performs a matrix-vector multiplication followed by an addition.
///
/// The function performs one of the matrix-vector operations
///
/// ```math
/// y := alpha * A * x + beta * y or
/// y := alpha * A^T * x + beta * y
/// ```
///
/// where `alpha` and `beta` are scalars, `x` and `y` are vectors, and `A` is
/// an `m`-by-`n` matrix.
///
/// http://www.netlib.org/lapack/explore-html/dc/da8/dgemv_8f.html
#[inline]
pub fn dgemv(trans: u8, m: uint, n: uint, alpha: f64, a: *const f64, lda: uint,
             x: *const f64, incx: uint, beta: f64, y: *mut f64, incy: uint) {

    unsafe {
        dgemv_(&(trans as i8), &(m as i32), &(n as i32), &alpha, a, &(lda as i32),
               x, &(incx as i32), &beta, y, &(incy as i32));
    }
}

/// Performs a matrix-matrix multiplication followed by an addition.
///
/// The function performs one of the matrix-matrix operations
///
/// ```math
/// C := alpha * op(A) * op(B) + beta * C
/// ```
///
/// where `op(X)` is one of
///
/// ```math
/// op(X) = X or
/// op(X) = X^T,
/// ```
///
/// `alpha` and `beta` are scalars, and `A`, `B`, and `C` are matrices, with
/// `op(A)` an `m`-by-`k` matrix, `op(B)` a `k`-by-`n` matrix, and `C` an
/// `m`-by-`n` matrix.
///
/// http://www.netlib.org/lapack/explore-html/d7/d2b/dgemm_8f.html
#[inline]
pub fn dgemm(transa: u8, transb: u8, m: uint, n: uint, k: uint, alpha: f64, a: *const f64,
             lda: uint, b: *const f64, ldb: uint, beta: f64, c: *mut f64, ldc: uint) {

    unsafe {
        dgemm_(&(transa as i8), &(transb as i8), &(m as i32), &(n as i32), &(k as i32),
               &alpha, a, &(lda as i32), b, &(ldb as i32), &beta, c, &(ldc as i32));
    }
}

#[cfg(test)]
mod test {
    macro_rules! assert_equal(
        ($given:expr , $expected:expr) => ({
            assert_eq!($given.len(), $expected.len());
            for i in range(0u, $given.len()) {
                assert_eq!($given[i], $expected[i]);
            }
        });
    )

    #[test]
    fn dgemv() {
        let (m, n) = (2, 3);

        let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![6.0, 8.0];

        super::dgemv(b'N', m, n, 1.0, a.as_ptr(), m, x.as_ptr(), 1,
                     1.0, y.as_mut_ptr(), 1);

        let expected_y = vec![20.0, 40.0];
        assert_equal!(y, expected_y);
    }

    #[test]
    fn dgemm() {
        let (m, n, k) = (2, 4, 3);

        let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let b = vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0];
        let mut c = vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0];

        super::dgemm(b'N', b'N', m, n, k, 1.0, a.as_ptr(), m, b.as_ptr(), k,
                     1.0, c.as_mut_ptr(), m);

        let expected_c = vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0];
        assert_equal!(c, expected_c);
    }
}

#[cfg(test)]
mod bench {
    extern crate test;

    #[bench]
    fn dgemv_few_large(b: &mut test::Bencher) {
        let m = 1000;

        let a = Vec::from_elem(m * m, 1.0);
        let x = Vec::from_elem(m * 1, 1.0);
        let mut y = Vec::from_elem(m * 1, 1.0);

        b.iter(|| {
            super::dgemv(b'N', m, m, 1.0, a.as_ptr(), m, x.as_ptr(), 1,
                         1.0, y.as_mut_ptr(), 1)
        });
    }

    #[bench]
    fn dgemv_many_small(b: &mut test::Bencher) {
        let m = 20;

        let a = Vec::from_elem(m * m, 1.0);
        let x = Vec::from_elem(m * 1, 1.0);
        let mut y = Vec::from_elem(m * 1, 1.0);

        b.iter(|| {
            for _ in range(0u, 20000) {
                super::dgemv(b'N', m, m, 1.0, a.as_ptr(), m, x.as_ptr(), 1,
                             1.0, y.as_mut_ptr(), 1);
            }
        });
    }
}
