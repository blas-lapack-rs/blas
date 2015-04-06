//! Minimalistic wrappers for BLAS routines.

use libc::c_char;
use raw::{self, int};

pub enum Trans {
    /// Do not transponse.
    N = b'N' as isize,
    /// Transpose.
    T = b'T' as isize,
    /// Conjugate transpose.
    C = b'C' as isize,
}

#[inline]
pub fn dgemv(trans: Trans, m: usize, n: usize, alpha: f64, a: &[f64], lda: usize, x: &[f64],
             incx: usize, beta: f64, y: &mut [f64], incy: usize) {

    unsafe {
        raw::dgemv_(&(trans as c_char) as *const _,
                    &(m as int) as *const _,
                    &(n as int) as *const _,
                    &alpha as *const _,
                    a.as_ptr(),
                    &(lda as int) as *const _,
                    x.as_ptr(),
                    &(incx as int) as *const _,
                    &beta as *const _,
                    y.as_mut_ptr(),
                    &(incy as int) as *const _);
    }
}

#[inline]
pub fn dgemm(transa: Trans, transb: Trans, m: usize, n: usize, k: usize, alpha: f64, a: &[f64],
             lda: usize, b: &[f64], ldb: usize, beta: f64, c: &mut [f64], ldc: usize) {

    unsafe {
        raw::dgemm_(&(transa as c_char) as *const _,
                    &(transb as c_char) as *const _,
                    &(m as int) as *const _,
                    &(n as int) as *const _,
                    &(k as int) as *const _,
                    &alpha as *const _,
                    a.as_ptr(),
                    &(lda as int) as *const _,
                    b.as_ptr(),
                    &(ldb as int) as *const _,
                    &beta as *const _,
                    c.as_mut_ptr(),
                    &(ldc as int) as *const _);
    }
}
