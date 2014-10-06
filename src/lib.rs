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

pub static NORMAL: i8 = 'N' as i8;
pub static TRANSPOSED: i8 = 'T' as i8;

#[inline]
pub fn dgemv(trans: i8, m: i32, n: i32, alpha: f64, a: *const f64,
    lda: i32, x: *const f64, incx: i32, beta: f64, y: *mut f64, incy: i32) {

    unsafe {
        dgemv_(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }
}

#[inline]
pub fn dgemm(transa: i8, transb: i8, m: i32, n: i32, k: i32, alpha: f64, a: *const f64,
    lda: i32, b: *const f64, ldb: i32, beta: f64, c: *mut f64, ldc: i32) {

    unsafe {
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }
}
