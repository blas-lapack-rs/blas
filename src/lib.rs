//! The library provides an interface to the [Basic Linear Algebra Subprograms](
//! http://www.netlib.org/blas/).

extern crate libc;

use libc::{c_char, c_double, c_int};

#[link(name = "gfortran")]
#[link(name = "blas", kind = "static")]
extern {
    fn dgemv_(trans: *const c_char, m: *const c_int, n: *const c_int,
              alpha: *const c_double, a: *const c_double, lda: *const c_int,
              x: *const c_double, incx: *const c_int, beta: *const c_double,
              y: *mut c_double, incy: *const c_int);

    fn dgemm_(transa: *const c_char, transb: *const c_char, m: *const c_int,
              n: *const c_int, k: *const c_int, alpha: *const c_double,
              a: *const c_double, lda: *const c_int, b: *const c_double,
              ldb: *const c_int, beta: *const c_double, c: *mut c_double,
              ldc: *const c_int);
}

/// Performs one of the matrix-vector operations
///
/// ```ignore
/// y := alpha * A * x + beta * y or
/// y := alpha * A**T * x + beta * y
/// ```
///
/// where alpha and beta are scalars, x and y are vectors, and A is an m-by-n
/// matrix.
///
/// http://www.netlib.org/lapack/explore-html/dc/da8/dgemv_8f.html
#[inline]
pub fn dgemv(trans: u8, m: uint, n: uint, alpha: f64, a: *const f64, lda: uint,
             x: *const f64, incx: uint, beta: f64, y: *mut f64, incy: uint) {

    unsafe {
        dgemv_(&(trans as i8), &(m as i32), &(n as i32), &alpha, a,
               &(lda as i32), x, &(incx as i32), &beta, y, &(incy as i32));
    }
}

/// Performs one of the matrix-matrix operations
///
/// ```ignore
/// C := alpha * op(A) * op(B) + beta * C
/// ```
///
/// where op(X) is one of
///
/// ```ignore
/// op(X) = X or
/// op(X) = X**T,
/// ```
///
/// alpha and beta are scalars, and A, B, and C are matrices, with op(A)
/// an m-by-k matrix, op(B) a k-by-n matrix, and C an m-by-n matrix.
///
/// http://www.netlib.org/lapack/explore-html/d7/d2b/dgemm_8f.html
#[inline]
pub fn dgemm(transa: u8, transb: u8, m: uint, n: uint, k: uint, alpha: f64,
             a: *const f64, lda: uint, b: *const f64, ldb: uint, beta: f64,
             c: *mut f64, ldc: uint) {

    unsafe {
        dgemm_(&(transa as i8), &(transb as i8), &(m as i32), &(n as i32),
               &(k as i32), &alpha, a, &(lda as i32), b, &(ldb as i32), &beta,
               c, &(ldc as i32));
    }
}
