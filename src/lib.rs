//! Interface to the [Basic Linear Algebra Subprograms][1].
//!
//! [1]: http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms

extern crate blas_sys as ffi;
extern crate complex;
extern crate libc;

use complex::{c32, c64};
use libc::{c_char, c_int};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Diag {
    N = b'N' as isize,
    U = b'U' as isize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Side {
    L = b'L' as isize,
    R = b'R' as isize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Trans {
    N = b'N' as isize,
    T = b'T' as isize,
    C = b'C' as isize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Uplo {
    U = b'U' as isize,
    L = b'L' as isize,
}

#[inline]
pub fn srotg(a: &mut f32, b: &mut f32, c: &mut f32, s: &mut f32) {
    unsafe {
        ffi::srotg_(a, b, c, s)
    }
}

#[inline]
pub fn srotmg(d1: &mut f32, d2: &mut f32, x1: &mut f32, y1: f32, param: &mut [f32]) {
    unsafe {
        ffi::srotmg_(d1, d2, x1, &y1, param.as_mut_ptr())
    }
}

#[inline]
pub fn srot(n: usize, x: &mut [f32], incx: usize, y: &mut [f32], incy: usize, c: f32, s: f32) {
    unsafe {
        ffi::srot_(&(n as c_int), x.as_mut_ptr(), &(incx as c_int), y.as_mut_ptr(),
                   &(incy as c_int), &c, &s)
    }
}

#[inline]
pub fn srotm(n: usize, x: &mut [f32], incx: usize, y: &mut [f32], incy: usize, param: &[f32]) {
    unsafe {
        ffi::srotm_(&(n as c_int), x.as_mut_ptr(), &(incx as c_int), y.as_mut_ptr(),
                    &(incy as c_int), param.as_ptr())
    }
}

#[inline]
pub fn sswap(n: usize, x: &mut [f32], incx: usize, y: &mut [f32], incy: usize) {
    unsafe {
        ffi::sswap_(&(n as c_int), x.as_mut_ptr(), &(incx as c_int), y.as_mut_ptr(),
                    &(incy as c_int))
    }
}

#[inline]
pub fn sscal(n: usize, a: f32, x: &mut [f32], incx: usize) {
    unsafe {
        ffi::sscal_(&(n as c_int), &a, x.as_mut_ptr(), &(incx as c_int))
    }
}

#[inline]
pub fn scopy(n: usize, x: &[f32], incx: usize, y: &mut [f32], incy: usize) {
    unsafe {
        ffi::scopy_(&(n as c_int), x.as_ptr(), &(incx as c_int), y.as_mut_ptr(), &(incy as c_int))
    }
}

#[inline]
pub fn saxpy(n: usize, alpha: f32, x: &[f32], incx: usize, y: &mut [f32], incy: usize) {
    unsafe {
        ffi::saxpy_(&(n as c_int), &alpha, x.as_ptr(), &(incx as c_int), y.as_mut_ptr(),
                    &(incy as c_int))
    }
}

#[inline]
pub fn sdot(n: usize, x: &[f32], incx: usize, y: &[f32], incy: usize) -> f32 {
    unsafe {
        ffi::sdot_(&(n as c_int), x.as_ptr(), &(incx as c_int), y.as_ptr(),
                   &(incy as c_int)) as f32
    }
}

#[inline]
pub fn sdsdot(n: usize, sb: &[f32], x: &[f32], incx: usize, y: &[f32], incy: usize) -> f32 {
    unsafe {
        ffi::sdsdot_(&(n as c_int), sb.as_ptr(), x.as_ptr(), &(incx as c_int), y.as_ptr(),
                     &(incy as c_int)) as f32
    }
}

#[inline]
pub fn snrm2(n: usize, x: &[f32], incx: usize) -> f32 {
    unsafe {
        ffi::snrm2_(&(n as c_int), x.as_ptr(), &(incx as c_int)) as f32
    }
}

#[inline]
pub fn scnrm2(n: usize, x: &[c32], incx: usize) -> f32 {
    unsafe {
        ffi::scnrm2_(&(n as c_int), x.as_ptr() as *const _, &(incx as c_int)) as f32
    }
}

#[inline]
pub fn sasum(n: usize, x: &[f32], incx: usize) -> f32 {
    unsafe {
        ffi::sasum_(&(n as c_int), x.as_ptr(), &(incx as c_int)) as f32
    }
}

#[inline]
pub fn isamax(n: usize, x: &[f32], incx: usize) -> isize {
    unsafe {
        ffi::isamax_(&(n as c_int), x.as_ptr(), &(incx as c_int)) as isize
    }
}

#[inline]
pub fn drotg(a: &mut f64, b: &mut f64, c: &mut f64, s: &mut f64) {
    unsafe {
        ffi::drotg_(a, b, c, s)
    }
}

#[inline]
pub fn drotmg(d1: &mut f64, d2: &mut f64, x1: &mut f64, y1: f64, param: &mut [f64]) {
    unsafe {
        ffi::drotmg_(d1, d2, x1, &y1, param.as_mut_ptr())
    }
}

#[inline]
pub fn drot(n: usize, x: &mut [f64], incx: usize, y: &mut [f64], incy: usize, c: f64, s: f64) {
    unsafe {
        ffi::drot_(&(n as c_int), x.as_mut_ptr(), &(incx as c_int), y.as_mut_ptr(),
                   &(incy as c_int), &c, &s)
    }
}

#[inline]
pub fn drotm(n: usize, x: &mut [f64], incx: usize, y: &mut [f64], incy: usize, param: &[f64]) {
    unsafe {
        ffi::drotm_(&(n as c_int), x.as_mut_ptr(), &(incx as c_int), y.as_mut_ptr(),
                    &(incy as c_int), param.as_ptr())
    }
}

#[inline]
pub fn dswap(n: usize, x: &mut [f64], incx: usize, y: &mut [f64], incy: usize) {
    unsafe {
        ffi::dswap_(&(n as c_int), x.as_mut_ptr(), &(incx as c_int), y.as_mut_ptr(),
                    &(incy as c_int))
    }
}

#[inline]
pub fn dscal(n: usize, a: f64, x: &mut [f64], incx: usize) {
    unsafe {
        ffi::dscal_(&(n as c_int), &a, x.as_mut_ptr(), &(incx as c_int))
    }
}

#[inline]
pub fn dcopy(n: usize, x: &[f64], incx: usize, y: &mut [f64], incy: usize) {
    unsafe {
        ffi::dcopy_(&(n as c_int), x.as_ptr(), &(incx as c_int), y.as_mut_ptr(), &(incy as c_int))
    }
}

#[inline]
pub fn daxpy(n: usize, alpha: f64, x: &[f64], incx: usize, y: &mut [f64], incy: usize) {
    unsafe {
        ffi::daxpy_(&(n as c_int), &alpha, x.as_ptr(), &(incx as c_int), y.as_mut_ptr(),
                    &(incy as c_int))
    }
}

#[inline]
pub fn ddot(n: usize, x: &[f64], incx: usize, y: &[f64], incy: usize) -> f64 {
    unsafe {
        ffi::ddot_(&(n as c_int), x.as_ptr(), &(incx as c_int), y.as_ptr(),
                   &(incy as c_int)) as f64
    }
}

#[inline]
pub fn dsdot(n: usize, x: &[f32], incx: usize, y: &[f32], incy: usize) -> f64 {
    unsafe {
        ffi::dsdot_(&(n as c_int), x.as_ptr(), &(incx as c_int), y.as_ptr(),
                    &(incy as c_int)) as f64
    }
}

#[inline]
pub fn dnrm2(n: usize, x: &[f64], incx: usize) -> f64 {
    unsafe {
        ffi::dnrm2_(&(n as c_int), x.as_ptr(), &(incx as c_int)) as f64
    }
}

#[inline]
pub fn dznrm2(n: usize, x: &[c64], incx: usize) -> f64 {
    unsafe {
        ffi::dznrm2_(&(n as c_int), x.as_ptr() as *const _, &(incx as c_int)) as f64
    }
}

#[inline]
pub fn dasum(n: usize, x: &[f64], incx: usize) -> f64 {
    unsafe {
        ffi::dasum_(&(n as c_int), x.as_ptr(), &(incx as c_int)) as f64
    }
}

#[inline]
pub fn idamax(n: usize, x: &[f64], incx: usize) -> isize {
    unsafe {
        ffi::idamax_(&(n as c_int), x.as_ptr(), &(incx as c_int)) as isize
    }
}

#[inline]
pub fn crotg(a: &mut c32, b: c32, c: &mut f32, s: &mut c32) {
    unsafe {
        ffi::crotg_(a as *mut _ as *mut _, &b as *const _ as *const _, c, s as *mut _ as *mut _)
    }
}

#[inline]
pub fn csrot(n: usize, x: &mut [c32], incx: usize, y: &mut [c32], incy: usize, c: f32, s: f32) {
    unsafe {
        ffi::csrot_(&(n as c_int), x.as_mut_ptr() as *mut _, &(incx as c_int),
                    y.as_mut_ptr() as *mut _, &(incy as c_int), &c, &s)
    }
}

#[inline]
pub fn cswap(n: usize, x: &mut [c32], incx: usize, y: &mut [c32], incy: usize) {
    unsafe {
        ffi::cswap_(&(n as c_int), x.as_mut_ptr() as *mut _, &(incx as c_int),
                    y.as_mut_ptr() as *mut _, &(incy as c_int))
    }
}

#[inline]
pub fn cscal(n: usize, a: c32, x: &mut [c32], incx: usize) {
    unsafe {
        ffi::cscal_(&(n as c_int), &a as *const _ as *const _, x.as_mut_ptr() as *mut _,
                    &(incx as c_int))
    }
}

#[inline]
pub fn csscal(n: usize, a: f32, x: &mut [c32], incx: usize) {
    unsafe {
        ffi::csscal_(&(n as c_int), &a, x.as_mut_ptr() as *mut _, &(incx as c_int))
    }
}

#[inline]
pub fn ccopy(n: usize, x: &[c32], incx: usize, y: &mut [c32], incy: usize) {
    unsafe {
        ffi::ccopy_(&(n as c_int), x.as_ptr() as *const _, &(incx as c_int),
                    y.as_mut_ptr() as *mut _, &(incy as c_int))
    }
}

#[inline]
pub fn caxpy(n: usize, alpha: c32, x: &[c32], incx: usize, y: &mut [c32], incy: usize) {
    unsafe {
        ffi::caxpy_(&(n as c_int), &alpha as *const _ as *const _, x.as_ptr() as *const _,
                    &(incx as c_int), y.as_mut_ptr() as *mut _, &(incy as c_int))
    }
}

#[inline]
pub fn cdotu(pres: &mut [c32], n: usize, x: &[c32], incx: usize, y: &[c32], incy: usize) {
    unsafe {
        ffi::cdotu_(pres.as_mut_ptr() as *mut _, &(n as c_int), x.as_ptr() as *const _,
                    &(incx as c_int), y.as_ptr() as *const _, &(incy as c_int))
    }
}

#[inline]
pub fn cdotc(pres: &mut [c32], n: usize, x: &[c32], incx: usize, y: &[c32], incy: usize) {
    unsafe {
        ffi::cdotc_(pres.as_mut_ptr() as *mut _, &(n as c_int), x.as_ptr() as *const _,
                    &(incx as c_int), y.as_ptr() as *const _, &(incy as c_int))
    }
}

#[inline]
pub fn scasum(n: usize, x: &[c32], incx: usize) -> f32 {
    unsafe {
        ffi::scasum_(&(n as c_int), x.as_ptr() as *const _, &(incx as c_int)) as f32
    }
}

#[inline]
pub fn icamax(n: usize, x: &[c32], incx: usize) -> isize {
    unsafe {
        ffi::icamax_(&(n as c_int), x.as_ptr() as *const _, &(incx as c_int)) as isize
    }
}

#[inline]
pub fn zrotg(a: &mut c64, b: c64, c: &mut f64, s: &mut c64) {
    unsafe {
        ffi::zrotg_(a as *mut _ as *mut _, &b as *const _ as *const _, c, s as *mut _ as *mut _)
    }
}

#[inline]
pub fn zdrot(n: usize, x: &mut [c64], incx: usize, y: &mut [c64], incy: usize, c: f64, s: f64) {
    unsafe {
        ffi::zdrot_(&(n as c_int), x.as_mut_ptr() as *mut _, &(incx as c_int),
                    y.as_mut_ptr() as *mut _, &(incy as c_int), &c, &s)
    }
}

#[inline]
pub fn zswap(n: usize, x: &mut [c64], incx: usize, y: &mut [c64], incy: usize) {
    unsafe {
        ffi::zswap_(&(n as c_int), x.as_mut_ptr() as *mut _, &(incx as c_int),
                    y.as_mut_ptr() as *mut _, &(incy as c_int))
    }
}

#[inline]
pub fn zscal(n: usize, a: c64, x: &mut [c64], incx: usize) {
    unsafe {
        ffi::zscal_(&(n as c_int), &a as *const _ as *const _, x.as_mut_ptr() as *mut _,
                    &(incx as c_int))
    }
}

#[inline]
pub fn zdscal(n: usize, a: f64, x: &mut [c64], incx: usize) {
    unsafe {
        ffi::zdscal_(&(n as c_int), &a, x.as_mut_ptr() as *mut _, &(incx as c_int))
    }
}

#[inline]
pub fn zcopy(n: usize, x: &[c64], incx: usize, y: &mut [c64], incy: usize) {
    unsafe {
        ffi::zcopy_(&(n as c_int), x.as_ptr() as *const _, &(incx as c_int),
                    y.as_mut_ptr() as *mut _, &(incy as c_int))
    }
}

#[inline]
pub fn zaxpy(n: usize, alpha: c64, x: &[c64], incx: usize, y: &mut [c64], incy: usize) {
    unsafe {
        ffi::zaxpy_(&(n as c_int), &alpha as *const _ as *const _, x.as_ptr() as *const _,
                    &(incx as c_int), y.as_mut_ptr() as *mut _, &(incy as c_int))
    }
}

#[inline]
pub fn zdotu(pres: &mut [c64], n: usize, x: &[c64], incx: usize, y: &[c64], incy: usize) {
    unsafe {
        ffi::zdotu_(pres.as_mut_ptr() as *mut _, &(n as c_int), x.as_ptr() as *const _,
                    &(incx as c_int), y.as_ptr() as *const _, &(incy as c_int))
    }
}

#[inline]
pub fn zdotc(pres: &mut [c64], n: usize, x: &[c64], incx: usize, y: &[c64], incy: usize) {
    unsafe {
        ffi::zdotc_(pres.as_mut_ptr() as *mut _, &(n as c_int), x.as_ptr() as *const _,
                    &(incx as c_int), y.as_ptr() as *const _, &(incy as c_int))
    }
}

#[inline]
pub fn dzasum(n: usize, x: &[c64], incx: usize) -> f64 {
    unsafe {
        ffi::dzasum_(&(n as c_int), x.as_ptr() as *const _, &(incx as c_int)) as f64
    }
}

#[inline]
pub fn izamax(n: usize, x: &[c64], incx: usize) -> isize {
    unsafe {
        ffi::izamax_(&(n as c_int), x.as_ptr() as *const _, &(incx as c_int)) as isize
    }
}

#[inline]
pub fn sgemv(trans: Trans, m: usize, n: usize, alpha: f32, a: &[f32], lda: usize, x: &[f32],
             incx: usize, beta: f32, y: &mut [f32], incy: usize) {

    unsafe {
        ffi::sgemv_(&(trans as c_char), &(m as c_int), &(n as c_int), &alpha, a.as_ptr(),
                    &(lda as c_int), x.as_ptr(), &(incx as c_int), &beta, y.as_mut_ptr(),
                    &(incy as c_int))
    }
}

#[inline]
pub fn sgbmv(trans: Trans, m: usize, n: usize, kl: usize, ku: usize, alpha: f32, a: &[f32],
             lda: usize, x: &[f32], incx: usize, beta: f32, y: &mut [f32], incy: usize) {

    unsafe {
        ffi::sgbmv_(&(trans as c_char), &(m as c_int), &(n as c_int), &(kl as c_int),
                    &(ku as c_int), &alpha, a.as_ptr(), &(lda as c_int), x.as_ptr(),
                    &(incx as c_int), &beta, y.as_mut_ptr(), &(incy as c_int))
    }
}

#[inline]
pub fn ssymv(uplo: Uplo, n: usize, alpha: f32, a: &[f32], lda: usize, x: &[f32], incx: usize,
             beta: f32, y: &mut [f32], incy: usize) {

    unsafe {
        ffi::ssymv_(&(uplo as c_char), &(n as c_int), &alpha, a.as_ptr(), &(lda as c_int),
                    x.as_ptr(), &(incx as c_int), &beta, y.as_mut_ptr(), &(incy as c_int))
    }
}

#[inline]
pub fn ssbmv(uplo: Uplo, n: usize, k: usize, alpha: f32, a: &[f32], lda: usize, x: &[f32],
             incx: usize, beta: f32, y: &mut [f32], incy: usize) {

    unsafe {
        ffi::ssbmv_(&(uplo as c_char), &(n as c_int), &(k as c_int), &alpha, a.as_ptr(),
                    &(lda as c_int), x.as_ptr(), &(incx as c_int), &beta, y.as_mut_ptr(),
                    &(incy as c_int))
    }
}

#[inline]
pub fn sspmv(uplo: Uplo, n: usize, alpha: f32, ap: &[f32], x: &[f32], incx: usize, beta: f32,
             y: &mut [f32], incy: usize) {

    unsafe {
        ffi::sspmv_(&(uplo as c_char), &(n as c_int), &alpha, ap.as_ptr(), x.as_ptr(),
                    &(incx as c_int), &beta, y.as_mut_ptr(), &(incy as c_int))
    }
}

#[inline]
pub fn strmv(uplo: Uplo, transa: Trans, diag: Diag, n: usize, a: &[f32], lda: usize, b: &mut [f32],
             incx: usize) {

    unsafe {
        ffi::strmv_(&(uplo as c_char), &(transa as c_char), &(diag as c_char), &(n as c_int),
                    a.as_ptr(), &(lda as c_int), b.as_mut_ptr(), &(incx as c_int))
    }
}

#[inline]
pub fn stbmv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, k: usize, a: &[f32], lda: usize,
             x: &mut [f32], incx: usize) {

    unsafe {
        ffi::stbmv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &(n as c_int),
                    &(k as c_int), a.as_ptr(), &(lda as c_int), x.as_mut_ptr(), &(incx as c_int))
    }
}

#[inline]
pub fn stpmv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, ap: &[f32], x: &mut [f32],
             incx: usize) {

    unsafe {
        ffi::stpmv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &(n as c_int),
                    ap.as_ptr(), x.as_mut_ptr(), &(incx as c_int))
    }
}

#[inline]
pub fn strsv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, a: &[f32], lda: usize, x: &mut [f32],
             incx: usize) {

    unsafe {
        ffi::strsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &(n as c_int),
                    a.as_ptr(), &(lda as c_int), x.as_mut_ptr(), &(incx as c_int))
    }
}

#[inline]
pub fn stbsv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, k: usize, a: &[f32], lda: usize,
             x: &mut [f32], incx: usize) {

    unsafe {
        ffi::stbsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &(n as c_int),
                    &(k as c_int), a.as_ptr(), &(lda as c_int), x.as_mut_ptr(), &(incx as c_int))
    }
}

#[inline]
pub fn stpsv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, ap: &[f32], x: &mut [f32],
             incx: usize) {

    unsafe {
        ffi::stpsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &(n as c_int),
                    ap.as_ptr(), x.as_mut_ptr(), &(incx as c_int))
    }
}

#[inline]
pub fn sger(m: usize, n: usize, alpha: f32, x: &[f32], incx: usize, y: &[f32], incy: usize,
            a: &mut [f32], lda: usize) {

    unsafe {
        ffi::sger_(&(m as c_int), &(n as c_int), &alpha, x.as_ptr(), &(incx as c_int), y.as_ptr(),
                   &(incy as c_int), a.as_mut_ptr(), &(lda as c_int))
    }
}

#[inline]
pub fn ssyr(uplo: Uplo, n: usize, alpha: f32, x: &[f32], incx: usize, a: &mut [f32], lda: usize) {
    unsafe {
        ffi::ssyr_(&(uplo as c_char), &(n as c_int), &alpha, x.as_ptr(), &(incx as c_int),
                   a.as_mut_ptr(), &(lda as c_int))
    }
}

#[inline]
pub fn sspr(uplo: Uplo, n: usize, alpha: f32, x: &[f32], incx: usize, ap: &mut [f32]) {
    unsafe {
        ffi::sspr_(&(uplo as c_char), &(n as c_int), &alpha, x.as_ptr(), &(incx as c_int),
                   ap.as_mut_ptr())
    }
}

#[inline]
pub fn ssyr2(uplo: Uplo, n: usize, alpha: f32, x: &[f32], incx: usize, y: &[f32], incy: usize,
             a: &mut [f32], lda: usize) {

    unsafe {
        ffi::ssyr2_(&(uplo as c_char), &(n as c_int), &alpha, x.as_ptr(), &(incx as c_int),
                    y.as_ptr(), &(incy as c_int), a.as_mut_ptr(), &(lda as c_int))
    }
}

#[inline]
pub fn sspr2(uplo: Uplo, n: usize, alpha: f32, x: &[f32], incx: usize, y: &[f32], incy: usize,
             ap: &mut [f32]) {

    unsafe {
        ffi::sspr2_(&(uplo as c_char), &(n as c_int), &alpha, x.as_ptr(), &(incx as c_int),
                    y.as_ptr(), &(incy as c_int), ap.as_mut_ptr())
    }
}

#[inline]
pub fn dgemv(trans: Trans, m: usize, n: usize, alpha: f64, a: &[f64], lda: usize, x: &[f64],
             incx: usize, beta: f64, y: &mut [f64], incy: usize) {

    unsafe {
        ffi::dgemv_(&(trans as c_char), &(m as c_int), &(n as c_int), &alpha, a.as_ptr(),
                    &(lda as c_int), x.as_ptr(), &(incx as c_int), &beta, y.as_mut_ptr(),
                    &(incy as c_int))
    }
}

#[inline]
pub fn dgbmv(trans: Trans, m: usize, n: usize, kl: usize, ku: usize, alpha: f64, a: &[f64],
             lda: usize, x: &[f64], incx: usize, beta: f64, y: &mut [f64], incy: usize) {

    unsafe {
        ffi::dgbmv_(&(trans as c_char), &(m as c_int), &(n as c_int), &(kl as c_int),
                    &(ku as c_int), &alpha, a.as_ptr(), &(lda as c_int), x.as_ptr(),
                    &(incx as c_int), &beta, y.as_mut_ptr(), &(incy as c_int))
    }
}

#[inline]
pub fn dsymv(uplo: Uplo, n: usize, alpha: f64, a: &[f64], lda: usize, x: &[f64], incx: usize,
             beta: f64, y: &mut [f64], incy: usize) {

    unsafe {
        ffi::dsymv_(&(uplo as c_char), &(n as c_int), &alpha, a.as_ptr(), &(lda as c_int),
                    x.as_ptr(), &(incx as c_int), &beta, y.as_mut_ptr(), &(incy as c_int))
    }
}

#[inline]
pub fn dsbmv(uplo: Uplo, n: usize, k: usize, alpha: f64, a: &[f64], lda: usize, x: &[f64],
             incx: usize, beta: f64, y: &mut [f64], incy: usize) {

    unsafe {
        ffi::dsbmv_(&(uplo as c_char), &(n as c_int), &(k as c_int), &alpha, a.as_ptr(),
                    &(lda as c_int), x.as_ptr(), &(incx as c_int), &beta, y.as_mut_ptr(),
                    &(incy as c_int))
    }
}

#[inline]
pub fn dspmv(uplo: Uplo, n: usize, alpha: f64, ap: &[f64], x: &[f64], incx: usize, beta: f64,
             y: &mut [f64], incy: usize) {

    unsafe {
        ffi::dspmv_(&(uplo as c_char), &(n as c_int), &alpha, ap.as_ptr(), x.as_ptr(),
                    &(incx as c_int), &beta, y.as_mut_ptr(), &(incy as c_int))
    }
}

#[inline]
pub fn dtrmv(uplo: Uplo, transa: Trans, diag: Diag, n: usize, a: &[f64], lda: usize, b: &mut [f64],
             incx: usize) {

    unsafe {
        ffi::dtrmv_(&(uplo as c_char), &(transa as c_char), &(diag as c_char), &(n as c_int),
                    a.as_ptr(), &(lda as c_int), b.as_mut_ptr(), &(incx as c_int))
    }
}

#[inline]
pub fn dtbmv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, k: usize, a: &[f64], lda: usize,
             x: &mut [f64], incx: usize) {

    unsafe {
        ffi::dtbmv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &(n as c_int),
                    &(k as c_int), a.as_ptr(), &(lda as c_int), x.as_mut_ptr(), &(incx as c_int))
    }
}

#[inline]
pub fn dtpmv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, ap: &[f64], x: &mut [f64],
             incx: usize) {

    unsafe {
        ffi::dtpmv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &(n as c_int),
                    ap.as_ptr(), x.as_mut_ptr(), &(incx as c_int))
    }
}

#[inline]
pub fn dtrsv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, a: &[f64], lda: usize, x: &mut [f64],
             incx: usize) {

    unsafe {
        ffi::dtrsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &(n as c_int),
                    a.as_ptr(), &(lda as c_int), x.as_mut_ptr(), &(incx as c_int))
    }
}

#[inline]
pub fn dtbsv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, k: usize, a: &[f64], lda: usize,
             x: &mut [f64], incx: usize) {

    unsafe {
        ffi::dtbsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &(n as c_int),
                    &(k as c_int), a.as_ptr(), &(lda as c_int), x.as_mut_ptr(), &(incx as c_int))
    }
}

#[inline]
pub fn dtpsv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, ap: &[f64], x: &mut [f64],
             incx: usize) {

    unsafe {
        ffi::dtpsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &(n as c_int),
                    ap.as_ptr(), x.as_mut_ptr(), &(incx as c_int))
    }
}

#[inline]
pub fn dger(m: usize, n: usize, alpha: f64, x: &[f64], incx: usize, y: &[f64], incy: usize,
            a: &mut [f64], lda: usize) {

    unsafe {
        ffi::dger_(&(m as c_int), &(n as c_int), &alpha, x.as_ptr(), &(incx as c_int), y.as_ptr(),
                   &(incy as c_int), a.as_mut_ptr(), &(lda as c_int))
    }
}

#[inline]
pub fn dsyr(uplo: Uplo, n: usize, alpha: f64, x: &[f64], incx: usize, a: &mut [f64], lda: usize) {
    unsafe {
        ffi::dsyr_(&(uplo as c_char), &(n as c_int), &alpha, x.as_ptr(), &(incx as c_int),
                   a.as_mut_ptr(), &(lda as c_int))
    }
}

#[inline]
pub fn dspr(uplo: Uplo, n: usize, alpha: f64, x: &[f64], incx: usize, ap: &mut [f64]) {
    unsafe {
        ffi::dspr_(&(uplo as c_char), &(n as c_int), &alpha, x.as_ptr(), &(incx as c_int),
                   ap.as_mut_ptr())
    }
}

#[inline]
pub fn dsyr2(uplo: Uplo, n: usize, alpha: f64, x: &[f64], incx: usize, y: &[f64], incy: usize,
             a: &mut [f64], lda: usize) {

    unsafe {
        ffi::dsyr2_(&(uplo as c_char), &(n as c_int), &alpha, x.as_ptr(), &(incx as c_int),
                    y.as_ptr(), &(incy as c_int), a.as_mut_ptr(), &(lda as c_int))
    }
}

#[inline]
pub fn dspr2(uplo: Uplo, n: usize, alpha: f64, x: &[f64], incx: usize, y: &[f64], incy: usize,
             ap: &mut [f64]) {

    unsafe {
        ffi::dspr2_(&(uplo as c_char), &(n as c_int), &alpha, x.as_ptr(), &(incx as c_int),
                    y.as_ptr(), &(incy as c_int), ap.as_mut_ptr())
    }
}

#[inline]
pub fn cgemv(trans: Trans, m: usize, n: usize, alpha: c32, a: &[c32], lda: usize, x: &[c32],
             incx: usize, beta: c32, y: &mut [c32], incy: usize) {

    unsafe {
        ffi::cgemv_(&(trans as c_char), &(m as c_int), &(n as c_int),
                    &alpha as *const _ as *const _, a.as_ptr() as *const _, &(lda as c_int),
                    x.as_ptr() as *const _, &(incx as c_int), &beta as *const _ as *const _,
                    y.as_mut_ptr() as *mut _, &(incy as c_int))
    }
}

#[inline]
pub fn cgbmv(trans: Trans, m: usize, n: usize, kl: usize, ku: usize, alpha: c32, a: &[c32],
             lda: usize, x: &[c32], incx: usize, beta: c32, y: &mut [c32], incy: usize) {

    unsafe {
        ffi::cgbmv_(&(trans as c_char), &(m as c_int), &(n as c_int), &(kl as c_int),
                    &(ku as c_int), &alpha as *const _ as *const _, a.as_ptr() as *const _,
                    &(lda as c_int), x.as_ptr() as *const _, &(incx as c_int),
                    &beta as *const _ as *const _, y.as_mut_ptr() as *mut _, &(incy as c_int))
    }
}

#[inline]
pub fn chemv(uplo: Uplo, n: usize, alpha: c32, a: &[c32], lda: usize, x: &[c32], incx: usize,
             beta: c32, y: &mut [c32], incy: usize) {

    unsafe {
        ffi::chemv_(&(uplo as c_char), &(n as c_int), &alpha as *const _ as *const _,
                    a.as_ptr() as *const _, &(lda as c_int), x.as_ptr() as *const _,
                    &(incx as c_int), &beta as *const _ as *const _, y.as_mut_ptr() as *mut _,
                    &(incy as c_int))
    }
}

#[inline]
pub fn chbmv(uplo: Uplo, n: usize, k: usize, alpha: c32, a: &[c32], lda: usize, x: &[c32],
             incx: usize, beta: c32, y: &mut [c32], incy: usize) {

    unsafe {
        ffi::chbmv_(&(uplo as c_char), &(n as c_int), &(k as c_int),
                    &alpha as *const _ as *const _, a.as_ptr() as *const _, &(lda as c_int),
                    x.as_ptr() as *const _, &(incx as c_int), &beta as *const _ as *const _,
                    y.as_mut_ptr() as *mut _, &(incy as c_int))
    }
}

#[inline]
pub fn chpmv(uplo: Uplo, n: usize, alpha: c32, ap: &[c32], x: &[c32], incx: usize, beta: c32,
             y: &mut [c32], incy: usize) {

    unsafe {
        ffi::chpmv_(&(uplo as c_char), &(n as c_int), &alpha as *const _ as *const _,
                    ap.as_ptr() as *const _, x.as_ptr() as *const _, &(incx as c_int),
                    &beta as *const _ as *const _, y.as_mut_ptr() as *mut _, &(incy as c_int))
    }
}

#[inline]
pub fn ctrmv(uplo: Uplo, transa: Trans, diag: Diag, n: usize, a: &[c32], lda: usize, b: &mut [c32],
             incx: usize) {

    unsafe {
        ffi::ctrmv_(&(uplo as c_char), &(transa as c_char), &(diag as c_char), &(n as c_int),
                    a.as_ptr() as *const _, &(lda as c_int), b.as_mut_ptr() as *mut _,
                    &(incx as c_int))
    }
}

#[inline]
pub fn ctbmv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, k: usize, a: &[c32], lda: usize,
             x: &mut [c32], incx: usize) {

    unsafe {
        ffi::ctbmv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &(n as c_int),
                    &(k as c_int), a.as_ptr() as *const _, &(lda as c_int),
                    x.as_mut_ptr() as *mut _, &(incx as c_int))
    }
}

#[inline]
pub fn ctpmv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, ap: &[c32], x: &mut [c32],
             incx: usize) {

    unsafe {
        ffi::ctpmv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &(n as c_int),
                    ap.as_ptr() as *const _, x.as_mut_ptr() as *mut _, &(incx as c_int))
    }
}

#[inline]
pub fn ctrsv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, a: &[c32], lda: usize, x: &mut [c32],
             incx: usize) {

    unsafe {
        ffi::ctrsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &(n as c_int),
                    a.as_ptr() as *const _, &(lda as c_int), x.as_mut_ptr() as *mut _,
                    &(incx as c_int))
    }
}

#[inline]
pub fn ctbsv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, k: usize, a: &[c32], lda: usize,
             x: &mut [c32], incx: usize) {

    unsafe {
        ffi::ctbsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &(n as c_int),
                    &(k as c_int), a.as_ptr() as *const _, &(lda as c_int),
                    x.as_mut_ptr() as *mut _, &(incx as c_int))
    }
}

#[inline]
pub fn ctpsv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, ap: &[c32], x: &mut [c32],
             incx: usize) {

    unsafe {
        ffi::ctpsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &(n as c_int),
                    ap.as_ptr() as *const _, x.as_mut_ptr() as *mut _, &(incx as c_int))
    }
}

#[inline]
pub fn cgeru(m: usize, n: usize, alpha: c32, x: &[c32], incx: usize, y: &[c32], incy: usize,
             a: &mut [c32], lda: usize) {

    unsafe {
        ffi::cgeru_(&(m as c_int), &(n as c_int), &alpha as *const _ as *const _,
                    x.as_ptr() as *const _, &(incx as c_int), y.as_ptr() as *const _,
                    &(incy as c_int), a.as_mut_ptr() as *mut _, &(lda as c_int))
    }
}

#[inline]
pub fn cgerc(m: usize, n: usize, alpha: c32, x: &[c32], incx: usize, y: &[c32], incy: usize,
             a: &mut [c32], lda: usize) {

    unsafe {
        ffi::cgerc_(&(m as c_int), &(n as c_int), &alpha as *const _ as *const _,
                    x.as_ptr() as *const _, &(incx as c_int), y.as_ptr() as *const _,
                    &(incy as c_int), a.as_mut_ptr() as *mut _, &(lda as c_int))
    }
}

#[inline]
pub fn cher(uplo: Uplo, n: usize, alpha: f32, x: &[c32], incx: usize, a: &mut [c32], lda: usize) {
    unsafe {
        ffi::cher_(&(uplo as c_char), &(n as c_int), &alpha, x.as_ptr() as *const _,
                   &(incx as c_int), a.as_mut_ptr() as *mut _, &(lda as c_int))
    }
}

#[inline]
pub fn chpr(uplo: Uplo, n: usize, alpha: f32, x: &[c32], incx: usize, ap: &mut [c32]) {
    unsafe {
        ffi::chpr_(&(uplo as c_char), &(n as c_int), &alpha, x.as_ptr() as *const _,
                   &(incx as c_int), ap.as_mut_ptr() as *mut _)
    }
}

#[inline]
pub fn chpr2(uplo: Uplo, n: usize, alpha: c32, x: &[c32], incx: usize, y: &[c32], incy: usize,
             ap: &mut [c32]) {

    unsafe {
        ffi::chpr2_(&(uplo as c_char), &(n as c_int), &alpha as *const _ as *const _,
                    x.as_ptr() as *const _, &(incx as c_int), y.as_ptr() as *const _,
                    &(incy as c_int), ap.as_mut_ptr() as *mut _)
    }
}

#[inline]
pub fn cher2(uplo: Uplo, n: usize, alpha: c32, x: &[c32], incx: usize, y: &[c32], incy: usize,
             a: &mut [c32], lda: usize) {

    unsafe {
        ffi::cher2_(&(uplo as c_char), &(n as c_int), &alpha as *const _ as *const _,
                    x.as_ptr() as *const _, &(incx as c_int), y.as_ptr() as *const _,
                    &(incy as c_int), a.as_mut_ptr() as *mut _, &(lda as c_int))
    }
}

#[inline]
pub fn zgemv(trans: Trans, m: usize, n: usize, alpha: c64, a: &[c64], lda: usize, x: &[c64],
             incx: usize, beta: c64, y: &mut [c64], incy: usize) {

    unsafe {
        ffi::zgemv_(&(trans as c_char), &(m as c_int), &(n as c_int),
                    &alpha as *const _ as *const _, a.as_ptr() as *const _, &(lda as c_int),
                    x.as_ptr() as *const _, &(incx as c_int), &beta as *const _ as *const _,
                    y.as_mut_ptr() as *mut _, &(incy as c_int))
    }
}

#[inline]
pub fn zgbmv(trans: Trans, m: usize, n: usize, kl: usize, ku: usize, alpha: c64, a: &[c64],
             lda: usize, x: &[c64], incx: usize, beta: c64, y: &mut [c64], incy: usize) {

    unsafe {
        ffi::zgbmv_(&(trans as c_char), &(m as c_int), &(n as c_int), &(kl as c_int),
                    &(ku as c_int), &alpha as *const _ as *const _, a.as_ptr() as *const _,
                    &(lda as c_int), x.as_ptr() as *const _, &(incx as c_int),
                    &beta as *const _ as *const _, y.as_mut_ptr() as *mut _, &(incy as c_int))
    }
}

#[inline]
pub fn zhemv(uplo: Uplo, n: usize, alpha: c64, a: &[c64], lda: usize, x: &[c64], incx: usize,
             beta: c64, y: &mut [c64], incy: usize) {

    unsafe {
        ffi::zhemv_(&(uplo as c_char), &(n as c_int), &alpha as *const _ as *const _,
                    a.as_ptr() as *const _, &(lda as c_int), x.as_ptr() as *const _,
                    &(incx as c_int), &beta as *const _ as *const _, y.as_mut_ptr() as *mut _,
                    &(incy as c_int))
    }
}

#[inline]
pub fn zhbmv(uplo: Uplo, n: usize, k: usize, alpha: c64, a: &[c64], lda: usize, x: &[c64],
             incx: usize, beta: c64, y: &mut [c64], incy: usize) {

    unsafe {
        ffi::zhbmv_(&(uplo as c_char), &(n as c_int), &(k as c_int),
                    &alpha as *const _ as *const _, a.as_ptr() as *const _, &(lda as c_int),
                    x.as_ptr() as *const _, &(incx as c_int), &beta as *const _ as *const _,
                    y.as_mut_ptr() as *mut _, &(incy as c_int))
    }
}

#[inline]
pub fn zhpmv(uplo: Uplo, n: usize, alpha: c64, ap: &[c64], x: &[c64], incx: usize, beta: c64,
             y: &mut [c64], incy: usize) {

    unsafe {
        ffi::zhpmv_(&(uplo as c_char), &(n as c_int), &alpha as *const _ as *const _,
                    ap.as_ptr() as *const _, x.as_ptr() as *const _, &(incx as c_int),
                    &beta as *const _ as *const _, y.as_mut_ptr() as *mut _, &(incy as c_int))
    }
}

#[inline]
pub fn ztrmv(uplo: Uplo, transa: Trans, diag: Diag, n: usize, a: &[c64], lda: usize, b: &mut [c64],
             incx: usize) {

    unsafe {
        ffi::ztrmv_(&(uplo as c_char), &(transa as c_char), &(diag as c_char), &(n as c_int),
                    a.as_ptr() as *const _, &(lda as c_int), b.as_mut_ptr() as *mut _,
                    &(incx as c_int))
    }
}

#[inline]
pub fn ztbmv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, k: usize, a: &[c64], lda: usize,
             x: &mut [c64], incx: usize) {

    unsafe {
        ffi::ztbmv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &(n as c_int),
                    &(k as c_int), a.as_ptr() as *const _, &(lda as c_int),
                    x.as_mut_ptr() as *mut _, &(incx as c_int))
    }
}

#[inline]
pub fn ztpmv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, ap: &[c64], x: &mut [c64],
             incx: usize) {

    unsafe {
        ffi::ztpmv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &(n as c_int),
                    ap.as_ptr() as *const _, x.as_mut_ptr() as *mut _, &(incx as c_int))
    }
}

#[inline]
pub fn ztrsv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, a: &[c64], lda: usize, x: &mut [c64],
             incx: usize) {

    unsafe {
        ffi::ztrsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &(n as c_int),
                    a.as_ptr() as *const _, &(lda as c_int), x.as_mut_ptr() as *mut _,
                    &(incx as c_int))
    }
}

#[inline]
pub fn ztbsv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, k: usize, a: &[c64], lda: usize,
             x: &mut [c64], incx: usize) {

    unsafe {
        ffi::ztbsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &(n as c_int),
                    &(k as c_int), a.as_ptr() as *const _, &(lda as c_int),
                    x.as_mut_ptr() as *mut _, &(incx as c_int))
    }
}

#[inline]
pub fn ztpsv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, ap: &[c64], x: &mut [c64],
             incx: usize) {

    unsafe {
        ffi::ztpsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &(n as c_int),
                    ap.as_ptr() as *const _, x.as_mut_ptr() as *mut _, &(incx as c_int))
    }
}

#[inline]
pub fn zgeru(m: usize, n: usize, alpha: c64, x: &[c64], incx: usize, y: &[c64], incy: usize,
             a: &mut [c64], lda: usize) {

    unsafe {
        ffi::zgeru_(&(m as c_int), &(n as c_int), &alpha as *const _ as *const _,
                    x.as_ptr() as *const _, &(incx as c_int), y.as_ptr() as *const _,
                    &(incy as c_int), a.as_mut_ptr() as *mut _, &(lda as c_int))
    }
}

#[inline]
pub fn zgerc(m: usize, n: usize, alpha: c64, x: &[c64], incx: usize, y: &[c64], incy: usize,
             a: &mut [c64], lda: usize) {

    unsafe {
        ffi::zgerc_(&(m as c_int), &(n as c_int), &alpha as *const _ as *const _,
                    x.as_ptr() as *const _, &(incx as c_int), y.as_ptr() as *const _,
                    &(incy as c_int), a.as_mut_ptr() as *mut _, &(lda as c_int))
    }
}

#[inline]
pub fn zher(uplo: Uplo, n: usize, alpha: f64, x: &[c64], incx: usize, a: &mut [c64], lda: usize) {
    unsafe {
        ffi::zher_(&(uplo as c_char), &(n as c_int), &alpha, x.as_ptr() as *const _,
                   &(incx as c_int), a.as_mut_ptr() as *mut _, &(lda as c_int))
    }
}

#[inline]
pub fn zhpr(uplo: Uplo, n: usize, alpha: f64, x: &[c64], incx: usize, ap: &mut [c64]) {
    unsafe {
        ffi::zhpr_(&(uplo as c_char), &(n as c_int), &alpha, x.as_ptr() as *const _,
                   &(incx as c_int), ap.as_mut_ptr() as *mut _)
    }
}

#[inline]
pub fn zher2(uplo: Uplo, n: usize, alpha: c64, x: &[c64], incx: usize, y: &[c64], incy: usize,
             a: &mut [c64], lda: usize) {

    unsafe {
        ffi::zher2_(&(uplo as c_char), &(n as c_int), &alpha as *const _ as *const _,
                    x.as_ptr() as *const _, &(incx as c_int), y.as_ptr() as *const _,
                    &(incy as c_int), a.as_mut_ptr() as *mut _, &(lda as c_int))
    }
}

#[inline]
pub fn zhpr2(uplo: Uplo, n: usize, alpha: c64, x: &[c64], incx: usize, y: &[c64], incy: usize,
             ap: &mut [c64]) {

    unsafe {
        ffi::zhpr2_(&(uplo as c_char), &(n as c_int), &alpha as *const _ as *const _,
                    x.as_ptr() as *const _, &(incx as c_int), y.as_ptr() as *const _,
                    &(incy as c_int), ap.as_mut_ptr() as *mut _)
    }
}

#[inline]
pub fn sgemm(transa: Trans, transb: Trans, m: usize, n: usize, k: usize, alpha: f32, a: &[f32],
             lda: usize, b: &[f32], ldb: usize, beta: f32, c: &mut [f32], ldc: usize) {

    unsafe {
        ffi::sgemm_(&(transa as c_char), &(transb as c_char), &(m as c_int), &(n as c_int),
                    &(k as c_int), &alpha, a.as_ptr(), &(lda as c_int), b.as_ptr(),
                    &(ldb as c_int), &beta, c.as_mut_ptr(), &(ldc as c_int))
    }
}

#[inline]
pub fn ssymm(side: Side, uplo: Uplo, m: usize, n: usize, alpha: f32, a: &[f32], lda: usize,
             b: &[f32], ldb: usize, beta: f32, c: &mut [f32], ldc: usize) {

    unsafe {
        ffi::ssymm_(&(side as c_char), &(uplo as c_char), &(m as c_int), &(n as c_int), &alpha,
                    a.as_ptr(), &(lda as c_int), b.as_ptr(), &(ldb as c_int), &beta,
                    c.as_mut_ptr(), &(ldc as c_int))
    }
}

#[inline]
pub fn ssyrk(uplo: Uplo, trans: Trans, n: usize, k: usize, alpha: f32, a: &[f32], lda: usize,
             beta: f32, c: &mut [f32], ldc: usize) {

    unsafe {
        ffi::ssyrk_(&(uplo as c_char), &(trans as c_char), &(n as c_int), &(k as c_int), &alpha,
                    a.as_ptr(), &(lda as c_int), &beta, c.as_mut_ptr(), &(ldc as c_int))
    }
}

#[inline]
pub fn ssyr2k(uplo: Uplo, trans: Trans, n: usize, k: usize, alpha: f32, a: &[f32], lda: usize,
              b: &[f32], ldb: usize, beta: f32, c: &mut [f32], ldc: usize) {

    unsafe {
        ffi::ssyr2k_(&(uplo as c_char), &(trans as c_char), &(n as c_int), &(k as c_int), &alpha,
                     a.as_ptr(), &(lda as c_int), b.as_ptr(), &(ldb as c_int), &beta,
                     c.as_mut_ptr(), &(ldc as c_int))
    }
}

#[inline]
pub fn strmm(side: Side, uplo: Uplo, transa: Trans, diag: Diag, m: usize, n: usize, alpha: f32,
             a: &[f32], lda: usize, b: &mut [f32], ldb: usize) {

    unsafe {
        ffi::strmm_(&(side as c_char), &(uplo as c_char), &(transa as c_char), &(diag as c_char),
                    &(m as c_int), &(n as c_int), &alpha, a.as_ptr(), &(lda as c_int),
                    b.as_mut_ptr(), &(ldb as c_int))
    }
}

#[inline]
pub fn strsm(side: Side, uplo: Uplo, transa: Trans, diag: Diag, m: usize, n: usize, alpha: f32,
             a: &[f32], lda: usize, b: &mut [f32], ldb: usize) {

    unsafe {
        ffi::strsm_(&(side as c_char), &(uplo as c_char), &(transa as c_char), &(diag as c_char),
                    &(m as c_int), &(n as c_int), &alpha, a.as_ptr(), &(lda as c_int),
                    b.as_mut_ptr(), &(ldb as c_int))
    }
}

#[inline]
pub fn dgemm(transa: Trans, transb: Trans, m: usize, n: usize, k: usize, alpha: f64, a: &[f64],
             lda: usize, b: &[f64], ldb: usize, beta: f64, c: &mut [f64], ldc: usize) {

    unsafe {
        ffi::dgemm_(&(transa as c_char), &(transb as c_char), &(m as c_int), &(n as c_int),
                    &(k as c_int), &alpha, a.as_ptr(), &(lda as c_int), b.as_ptr(),
                    &(ldb as c_int), &beta, c.as_mut_ptr(), &(ldc as c_int))
    }
}

#[inline]
pub fn dsymm(side: Side, uplo: Uplo, m: usize, n: usize, alpha: f64, a: &[f64], lda: usize,
             b: &[f64], ldb: usize, beta: f64, c: &mut [f64], ldc: usize) {

    unsafe {
        ffi::dsymm_(&(side as c_char), &(uplo as c_char), &(m as c_int), &(n as c_int), &alpha,
                    a.as_ptr(), &(lda as c_int), b.as_ptr(), &(ldb as c_int), &beta,
                    c.as_mut_ptr(), &(ldc as c_int))
    }
}

#[inline]
pub fn dsyrk(uplo: Uplo, trans: Trans, n: usize, k: usize, alpha: f64, a: &[f64], lda: usize,
             beta: f64, c: &mut [f64], ldc: usize) {

    unsafe {
        ffi::dsyrk_(&(uplo as c_char), &(trans as c_char), &(n as c_int), &(k as c_int), &alpha,
                    a.as_ptr(), &(lda as c_int), &beta, c.as_mut_ptr(), &(ldc as c_int))
    }
}

#[inline]
pub fn dsyr2k(uplo: Uplo, trans: Trans, n: usize, k: usize, alpha: f64, a: &[f64], lda: usize,
              b: &[f64], ldb: usize, beta: f64, c: &mut [f64], ldc: usize) {

    unsafe {
        ffi::dsyr2k_(&(uplo as c_char), &(trans as c_char), &(n as c_int), &(k as c_int), &alpha,
                     a.as_ptr(), &(lda as c_int), b.as_ptr(), &(ldb as c_int), &beta,
                     c.as_mut_ptr(), &(ldc as c_int))
    }
}

#[inline]
pub fn dtrmm(side: Side, uplo: Uplo, transa: Trans, diag: Diag, m: usize, n: usize, alpha: f64,
             a: &[f64], lda: usize, b: &mut [f64], ldb: usize) {

    unsafe {
        ffi::dtrmm_(&(side as c_char), &(uplo as c_char), &(transa as c_char), &(diag as c_char),
                    &(m as c_int), &(n as c_int), &alpha, a.as_ptr(), &(lda as c_int),
                    b.as_mut_ptr(), &(ldb as c_int))
    }
}

#[inline]
pub fn dtrsm(side: Side, uplo: Uplo, transa: Trans, diag: Diag, m: usize, n: usize, alpha: f64,
             a: &[f64], lda: usize, b: &mut [f64], ldb: usize) {

    unsafe {
        ffi::dtrsm_(&(side as c_char), &(uplo as c_char), &(transa as c_char), &(diag as c_char),
                    &(m as c_int), &(n as c_int), &alpha, a.as_ptr(), &(lda as c_int),
                    b.as_mut_ptr(), &(ldb as c_int))
    }
}

#[inline]
pub fn cgemm(transa: Trans, transb: Trans, m: usize, n: usize, k: usize, alpha: c32, a: &[c32],
             lda: usize, b: &[c32], ldb: usize, beta: c32, c: &mut [c32], ldc: usize) {

    unsafe {
        ffi::cgemm_(&(transa as c_char), &(transb as c_char), &(m as c_int), &(n as c_int),
                    &(k as c_int), &alpha as *const _ as *const _, a.as_ptr() as *const _,
                    &(lda as c_int), b.as_ptr() as *const _, &(ldb as c_int),
                    &beta as *const _ as *const _, c.as_mut_ptr() as *mut _, &(ldc as c_int))
    }
}

#[inline]
pub fn csymm(side: Side, uplo: Uplo, m: usize, n: usize, alpha: c32, a: &[c32], lda: usize,
             b: &[c32], ldb: usize, beta: c32, c: &mut [c32], ldc: usize) {

    unsafe {
        ffi::csymm_(&(side as c_char), &(uplo as c_char), &(m as c_int), &(n as c_int),
                    &alpha as *const _ as *const _, a.as_ptr() as *const _, &(lda as c_int),
                    b.as_ptr() as *const _, &(ldb as c_int), &beta as *const _ as *const _,
                    c.as_mut_ptr() as *mut _, &(ldc as c_int))
    }
}

#[inline]
pub fn chemm(side: Side, uplo: Uplo, m: usize, n: usize, alpha: c32, a: &[c32], lda: usize,
             b: &[c32], ldb: usize, beta: c32, c: &mut [c32], ldc: usize) {

    unsafe {
        ffi::chemm_(&(side as c_char), &(uplo as c_char), &(m as c_int), &(n as c_int),
                    &alpha as *const _ as *const _, a.as_ptr() as *const _, &(lda as c_int),
                    b.as_ptr() as *const _, &(ldb as c_int), &beta as *const _ as *const _,
                    c.as_mut_ptr() as *mut _, &(ldc as c_int))
    }
}

#[inline]
pub fn csyrk(uplo: Uplo, trans: Trans, n: usize, k: usize, alpha: c32, a: &[c32], lda: usize,
             beta: c32, c: &mut [c32], ldc: usize) {

    unsafe {
        ffi::csyrk_(&(uplo as c_char), &(trans as c_char), &(n as c_int), &(k as c_int),
                    &alpha as *const _ as *const _, a.as_ptr() as *const _, &(lda as c_int),
                    &beta as *const _ as *const _, c.as_mut_ptr() as *mut _, &(ldc as c_int))
    }
}

#[inline]
pub fn cherk(uplo: Uplo, trans: Trans, n: usize, k: usize, alpha: f32, a: &[c32], lda: usize,
             beta: f32, c: &mut [c32], ldc: usize) {

    unsafe {
        ffi::cherk_(&(uplo as c_char), &(trans as c_char), &(n as c_int), &(k as c_int), &alpha,
                    a.as_ptr() as *const _, &(lda as c_int), &beta, c.as_mut_ptr() as *mut _,
                    &(ldc as c_int))
    }
}

#[inline]
pub fn csyr2k(uplo: Uplo, trans: Trans, n: usize, k: usize, alpha: c32, a: &[c32], lda: usize,
              b: &[c32], ldb: usize, beta: c32, c: &mut [c32], ldc: usize) {

    unsafe {
        ffi::csyr2k_(&(uplo as c_char), &(trans as c_char), &(n as c_int), &(k as c_int),
                     &alpha as *const _ as *const _, a.as_ptr() as *const _, &(lda as c_int),
                     b.as_ptr() as *const _, &(ldb as c_int), &beta as *const _ as *const _,
                     c.as_mut_ptr() as *mut _, &(ldc as c_int))
    }
}

#[inline]
pub fn cher2k(uplo: Uplo, trans: Trans, n: usize, k: usize, alpha: c32, a: &[c32], lda: usize,
              b: &[c32], ldb: usize, beta: f32, c: &mut [c32], ldc: usize) {

    unsafe {
        ffi::cher2k_(&(uplo as c_char), &(trans as c_char), &(n as c_int), &(k as c_int),
                     &alpha as *const _ as *const _, a.as_ptr() as *const _, &(lda as c_int),
                     b.as_ptr() as *const _, &(ldb as c_int), &beta, c.as_mut_ptr() as *mut _,
                     &(ldc as c_int))
    }
}

#[inline]
pub fn ctrmm(side: Side, uplo: Uplo, transa: Trans, diag: Diag, m: usize, n: usize, alpha: c32,
             a: &[c32], lda: usize, b: &mut [c32], ldb: usize) {

    unsafe {
        ffi::ctrmm_(&(side as c_char), &(uplo as c_char), &(transa as c_char), &(diag as c_char),
                    &(m as c_int), &(n as c_int), &alpha as *const _ as *const _,
                    a.as_ptr() as *const _, &(lda as c_int), b.as_mut_ptr() as *mut _,
                    &(ldb as c_int))
    }
}

#[inline]
pub fn ctrsm(side: Side, uplo: Uplo, transa: Trans, diag: Diag, m: usize, n: usize, alpha: c32,
             a: &[c32], lda: usize, b: &mut [c32], ldb: usize) {

    unsafe {
        ffi::ctrsm_(&(side as c_char), &(uplo as c_char), &(transa as c_char), &(diag as c_char),
                    &(m as c_int), &(n as c_int), &alpha as *const _ as *const _,
                    a.as_ptr() as *const _, &(lda as c_int), b.as_mut_ptr() as *mut _,
                    &(ldb as c_int))
    }
}

#[inline]
pub fn zgemm(transa: Trans, transb: Trans, m: usize, n: usize, k: usize, alpha: c64, a: &[c64],
             lda: usize, b: &[c64], ldb: usize, beta: c64, c: &mut [c64], ldc: usize) {

    unsafe {
        ffi::zgemm_(&(transa as c_char), &(transb as c_char), &(m as c_int), &(n as c_int),
                    &(k as c_int), &alpha as *const _ as *const _, a.as_ptr() as *const _,
                    &(lda as c_int), b.as_ptr() as *const _, &(ldb as c_int),
                    &beta as *const _ as *const _, c.as_mut_ptr() as *mut _, &(ldc as c_int))
    }
}

#[inline]
pub fn zsymm(side: Side, uplo: Uplo, m: usize, n: usize, alpha: c64, a: &[c64], lda: usize,
             b: &[c64], ldb: usize, beta: c64, c: &mut [c64], ldc: usize) {

    unsafe {
        ffi::zsymm_(&(side as c_char), &(uplo as c_char), &(m as c_int), &(n as c_int),
                    &alpha as *const _ as *const _, a.as_ptr() as *const _, &(lda as c_int),
                    b.as_ptr() as *const _, &(ldb as c_int), &beta as *const _ as *const _,
                    c.as_mut_ptr() as *mut _, &(ldc as c_int))
    }
}

#[inline]
pub fn zhemm(side: Side, uplo: Uplo, m: usize, n: usize, alpha: c64, a: &[c64], lda: usize,
             b: &[c64], ldb: usize, beta: c64, c: &mut [c64], ldc: usize) {

    unsafe {
        ffi::zhemm_(&(side as c_char), &(uplo as c_char), &(m as c_int), &(n as c_int),
                    &alpha as *const _ as *const _, a.as_ptr() as *const _, &(lda as c_int),
                    b.as_ptr() as *const _, &(ldb as c_int), &beta as *const _ as *const _,
                    c.as_mut_ptr() as *mut _, &(ldc as c_int))
    }
}

#[inline]
pub fn zsyrk(uplo: Uplo, trans: Trans, n: usize, k: usize, alpha: c64, a: &[c64], lda: usize,
             beta: c64, c: &mut [c64], ldc: usize) {

    unsafe {
        ffi::zsyrk_(&(uplo as c_char), &(trans as c_char), &(n as c_int), &(k as c_int),
                    &alpha as *const _ as *const _, a.as_ptr() as *const _, &(lda as c_int),
                    &beta as *const _ as *const _, c.as_mut_ptr() as *mut _, &(ldc as c_int))
    }
}

#[inline]
pub fn zherk(uplo: Uplo, trans: Trans, n: usize, k: usize, alpha: f64, a: &[c64], lda: usize,
             beta: f64, c: &mut [c64], ldc: usize) {

    unsafe {
        ffi::zherk_(&(uplo as c_char), &(trans as c_char), &(n as c_int), &(k as c_int), &alpha,
                    a.as_ptr() as *const _, &(lda as c_int), &beta, c.as_mut_ptr() as *mut _,
                    &(ldc as c_int))
    }
}

#[inline]
pub fn zsyr2k(uplo: Uplo, trans: Trans, n: usize, k: usize, alpha: c64, a: &[c64], lda: usize,
              b: &[c64], ldb: usize, beta: c64, c: &mut [c64], ldc: usize) {

    unsafe {
        ffi::zsyr2k_(&(uplo as c_char), &(trans as c_char), &(n as c_int), &(k as c_int),
                     &alpha as *const _ as *const _, a.as_ptr() as *const _, &(lda as c_int),
                     b.as_ptr() as *const _, &(ldb as c_int), &beta as *const _ as *const _,
                     c.as_mut_ptr() as *mut _, &(ldc as c_int))
    }
}

#[inline]
pub fn zher2k(uplo: Uplo, trans: Trans, n: usize, k: usize, alpha: c64, a: &[c64], lda: usize,
              b: &[c64], ldb: usize, beta: f64, c: &mut [c64], ldc: usize) {

    unsafe {
        ffi::zher2k_(&(uplo as c_char), &(trans as c_char), &(n as c_int), &(k as c_int),
                     &alpha as *const _ as *const _, a.as_ptr() as *const _, &(lda as c_int),
                     b.as_ptr() as *const _, &(ldb as c_int), &beta, c.as_mut_ptr() as *mut _,
                     &(ldc as c_int))
    }
}

#[inline]
pub fn ztrmm(side: Side, uplo: Uplo, transa: Trans, diag: Diag, m: usize, n: usize, alpha: c64,
             a: &[c64], lda: usize, b: &mut [c64], ldb: usize) {

    unsafe {
        ffi::ztrmm_(&(side as c_char), &(uplo as c_char), &(transa as c_char), &(diag as c_char),
                    &(m as c_int), &(n as c_int), &alpha as *const _ as *const _,
                    a.as_ptr() as *const _, &(lda as c_int), b.as_mut_ptr() as *mut _,
                    &(ldb as c_int))
    }
}

#[inline]
pub fn ztrsm(side: Side, uplo: Uplo, transa: Trans, diag: Diag, m: usize, n: usize, alpha: c64,
             a: &[c64], lda: usize, b: &mut [c64], ldb: usize) {

    unsafe {
        ffi::ztrsm_(&(side as c_char), &(uplo as c_char), &(transa as c_char), &(diag as c_char),
                    &(m as c_int), &(n as c_int), &alpha as *const _ as *const _,
                    a.as_ptr() as *const _, &(lda as c_int), b.as_mut_ptr() as *mut _,
                    &(ldb as c_int))
    }
}
