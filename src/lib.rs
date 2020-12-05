//! Wrappers for [BLAS] \(Fortran).
//!
//! ## [Architecture]
//!
//! ## Example
//!
//! ```no_run
//! use blas::*;
//!
//! let (m, n, k) = (2, 4, 3);
//! let a = vec![
//!     1.0, 4.0,
//!     2.0, 5.0,
//!     3.0, 6.0,
//! ];
//! let b = vec![
//!     1.0, 5.0,  9.0,
//!     2.0, 6.0, 10.0,
//!     3.0, 7.0, 11.0,
//!     4.0, 8.0, 12.0,
//! ];
//! let mut c = vec![
//!     2.0, 7.0,
//!     6.0, 2.0,
//!     0.0, 7.0,
//!     4.0, 2.0,
//! ];
//!
//! unsafe {
//!     dgemm(b'N', b'N', m, n, k, 1.0, &a, m, &b, k, 1.0, &mut c, m);
//! }
//!
//! assert!(
//!     c == vec![
//!         40.0,  90.0,
//!         50.0, 100.0,
//!         50.0, 120.0,
//!         60.0, 130.0,
//!     ]
//! );
//! ```
//!
//! [architecture]: https://blas-lapack-rs.github.io/architecture
//! [blas]: https://en.wikipedia.org/wiki/BLAS

extern crate blas_sys as ffi;
extern crate libc;
extern crate num_complex as num;

use libc::c_char;

/// A complex number with 32-bit parts.
#[allow(non_camel_case_types)]
pub type c32 = num::Complex<f32>;

/// A complex number with 64-bit parts.
#[allow(non_camel_case_types)]
pub type c64 = num::Complex<f64>;

#[inline]
pub unsafe fn srotg(a: &mut f32, b: &mut f32, c: &mut f32, s: &mut f32) {
    ffi::srotg_(a, b, c, s)
}

#[inline]
pub unsafe fn srotmg(d1: &mut f32, d2: &mut f32, x1: &mut f32, y1: f32, param: &mut [f32]) {
    ffi::srotmg_(d1, d2, x1, &y1, param.as_mut_ptr())
}

#[inline]
pub unsafe fn srot(n: i32, x: &mut [f32], incx: i32, y: &mut [f32], incy: i32, c: f32, s: f32) {
    ffi::srot_(&n, x.as_mut_ptr(), &incx, y.as_mut_ptr(), &incy, &c, &s)
}

#[inline]
pub unsafe fn srotm(n: i32, x: &mut [f32], incx: i32, y: &mut [f32], incy: i32, param: &[f32]) {
    ffi::srotm_(
        &n,
        x.as_mut_ptr(),
        &incx,
        y.as_mut_ptr(),
        &incy,
        param.as_ptr(),
    )
}

#[inline]
pub unsafe fn sswap(n: i32, x: &mut [f32], incx: i32, y: &mut [f32], incy: i32) {
    ffi::sswap_(&n, x.as_mut_ptr(), &incx, y.as_mut_ptr(), &incy)
}

#[inline]
pub unsafe fn sscal(n: i32, a: f32, x: &mut [f32], incx: i32) {
    ffi::sscal_(&n, &a, x.as_mut_ptr(), &incx)
}

#[inline]
pub unsafe fn scopy(n: i32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
    ffi::scopy_(&n, x.as_ptr(), &incx, y.as_mut_ptr(), &incy)
}

#[inline]
pub unsafe fn saxpy(n: i32, alpha: f32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
    ffi::saxpy_(&n, &alpha, x.as_ptr(), &incx, y.as_mut_ptr(), &incy)
}

#[inline]
pub unsafe fn sdot(n: i32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
    ffi::sdot_(&n, x.as_ptr(), &incx, y.as_ptr(), &incy)
}

#[inline]
pub unsafe fn sdsdot(n: i32, sb: &[f32], x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
    ffi::sdsdot_(&n, sb.as_ptr(), x.as_ptr(), &incx, y.as_ptr(), &incy)
}

#[inline]
pub unsafe fn snrm2(n: i32, x: &[f32], incx: i32) -> f32 {
    ffi::snrm2_(&n, x.as_ptr(), &incx)
}

#[inline]
pub unsafe fn scnrm2(n: i32, x: &[c32], incx: i32) -> f32 {
    ffi::scnrm2_(&n, x.as_ptr() as *const _, &incx)
}

#[inline]
pub unsafe fn sasum(n: i32, x: &[f32], incx: i32) -> f32 {
    ffi::sasum_(&n, x.as_ptr(), &incx)
}

#[inline]
pub unsafe fn isamax(n: i32, x: &[f32], incx: i32) -> usize {
    ffi::isamax_(&n, x.as_ptr(), &incx) as usize
}

#[inline]
pub unsafe fn drotg(a: &mut f64, b: &mut f64, c: &mut f64, s: &mut f64) {
    ffi::drotg_(a, b, c, s)
}

#[inline]
pub unsafe fn drotmg(d1: &mut f64, d2: &mut f64, x1: &mut f64, y1: f64, param: &mut [f64]) {
    ffi::drotmg_(d1, d2, x1, &y1, param.as_mut_ptr())
}

#[inline]
pub unsafe fn drot(n: i32, x: &mut [f64], incx: i32, y: &mut [f64], incy: i32, c: f64, s: f64) {
    ffi::drot_(&n, x.as_mut_ptr(), &incx, y.as_mut_ptr(), &incy, &c, &s)
}

#[inline]
pub unsafe fn drotm(n: i32, x: &mut [f64], incx: i32, y: &mut [f64], incy: i32, param: &[f64]) {
    ffi::drotm_(
        &n,
        x.as_mut_ptr(),
        &incx,
        y.as_mut_ptr(),
        &incy,
        param.as_ptr(),
    )
}

#[inline]
pub unsafe fn dswap(n: i32, x: &mut [f64], incx: i32, y: &mut [f64], incy: i32) {
    ffi::dswap_(&n, x.as_mut_ptr(), &incx, y.as_mut_ptr(), &incy)
}

#[inline]
pub unsafe fn dscal(n: i32, a: f64, x: &mut [f64], incx: i32) {
    ffi::dscal_(&n, &a, x.as_mut_ptr(), &incx)
}

#[inline]
pub unsafe fn dcopy(n: i32, x: &[f64], incx: i32, y: &mut [f64], incy: i32) {
    ffi::dcopy_(&n, x.as_ptr(), &incx, y.as_mut_ptr(), &incy)
}

#[inline]
pub unsafe fn daxpy(n: i32, alpha: f64, x: &[f64], incx: i32, y: &mut [f64], incy: i32) {
    ffi::daxpy_(&n, &alpha, x.as_ptr(), &incx, y.as_mut_ptr(), &incy)
}

#[inline]
pub unsafe fn ddot(n: i32, x: &[f64], incx: i32, y: &[f64], incy: i32) -> f64 {
    ffi::ddot_(&n, x.as_ptr(), &incx, y.as_ptr(), &incy)
}

#[inline]
pub unsafe fn dsdot(n: i32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f64 {
    ffi::dsdot_(&n, x.as_ptr(), &incx, y.as_ptr(), &incy)
}

#[inline]
pub unsafe fn dnrm2(n: i32, x: &[f64], incx: i32) -> f64 {
    ffi::dnrm2_(&n, x.as_ptr(), &incx)
}

#[inline]
pub unsafe fn dznrm2(n: i32, x: &[c64], incx: i32) -> f64 {
    ffi::dznrm2_(&n, x.as_ptr() as *const _, &incx)
}

#[inline]
pub unsafe fn dasum(n: i32, x: &[f64], incx: i32) -> f64 {
    ffi::dasum_(&n, x.as_ptr(), &incx)
}

#[inline]
pub unsafe fn idamax(n: i32, x: &[f64], incx: i32) -> usize {
    ffi::idamax_(&n, x.as_ptr(), &incx) as usize
}

#[inline]
pub unsafe fn crotg(a: &mut c32, b: c32, c: &mut f32, s: &mut c32) {
    ffi::crotg_(
        a as *mut _ as *mut _,
        &b as *const _ as *const _,
        c,
        s as *mut _ as *mut _,
    )
}

#[inline]
pub unsafe fn csrot(n: i32, x: &mut [c32], incx: i32, y: &mut [c32], incy: i32, c: f32, s: f32) {
    ffi::csrot_(
        &n,
        x.as_mut_ptr() as *mut _,
        &incx,
        y.as_mut_ptr() as *mut _,
        &incy,
        &c,
        &s,
    )
}

#[inline]
pub unsafe fn cswap(n: i32, x: &mut [c32], incx: i32, y: &mut [c32], incy: i32) {
    ffi::cswap_(
        &n,
        x.as_mut_ptr() as *mut _,
        &incx,
        y.as_mut_ptr() as *mut _,
        &incy,
    )
}

#[inline]
pub unsafe fn cscal(n: i32, a: c32, x: &mut [c32], incx: i32) {
    ffi::cscal_(
        &n,
        &a as *const _ as *const _,
        x.as_mut_ptr() as *mut _,
        &incx,
    )
}

#[inline]
pub unsafe fn csscal(n: i32, a: f32, x: &mut [c32], incx: i32) {
    ffi::csscal_(&n, &a, x.as_mut_ptr() as *mut _, &incx)
}

#[inline]
pub unsafe fn ccopy(n: i32, x: &[c32], incx: i32, y: &mut [c32], incy: i32) {
    ffi::ccopy_(
        &n,
        x.as_ptr() as *const _,
        &incx,
        y.as_mut_ptr() as *mut _,
        &incy,
    )
}

#[inline]
pub unsafe fn caxpy(n: i32, alpha: c32, x: &[c32], incx: i32, y: &mut [c32], incy: i32) {
    ffi::caxpy_(
        &n,
        &alpha as *const _ as *const _,
        x.as_ptr() as *const _,
        &incx,
        y.as_mut_ptr() as *mut _,
        &incy,
    )
}

#[inline]
pub unsafe fn cdotu(pres: &mut [c32], n: i32, x: &[c32], incx: i32, y: &[c32], incy: i32) {
    ffi::cdotu_(
        pres.as_mut_ptr() as *mut _,
        &n,
        x.as_ptr() as *const _,
        &incx,
        y.as_ptr() as *const _,
        &incy,
    )
}

#[inline]
pub unsafe fn cdotc(pres: &mut [c32], n: i32, x: &[c32], incx: i32, y: &[c32], incy: i32) {
    ffi::cdotc_(
        pres.as_mut_ptr() as *mut _,
        &n,
        x.as_ptr() as *const _,
        &incx,
        y.as_ptr() as *const _,
        &incy,
    )
}

#[inline]
pub unsafe fn scasum(n: i32, x: &[c32], incx: i32) -> f32 {
    ffi::scasum_(&n, x.as_ptr() as *const _, &incx)
}

#[inline]
pub unsafe fn icamax(n: i32, x: &[c32], incx: i32) -> usize {
    ffi::icamax_(&n, x.as_ptr() as *const _, &incx) as usize
}

#[inline]
pub unsafe fn zrotg(a: &mut c64, b: c64, c: &mut f64, s: &mut c64) {
    ffi::zrotg_(
        a as *mut _ as *mut _,
        &b as *const _ as *const _,
        c,
        s as *mut _ as *mut _,
    )
}

#[inline]
pub unsafe fn zdrot(n: i32, x: &mut [c64], incx: i32, y: &mut [c64], incy: i32, c: f64, s: f64) {
    ffi::zdrot_(
        &n,
        x.as_mut_ptr() as *mut _,
        &incx,
        y.as_mut_ptr() as *mut _,
        &incy,
        &c,
        &s,
    )
}

#[inline]
pub unsafe fn zswap(n: i32, x: &mut [c64], incx: i32, y: &mut [c64], incy: i32) {
    ffi::zswap_(
        &n,
        x.as_mut_ptr() as *mut _,
        &incx,
        y.as_mut_ptr() as *mut _,
        &incy,
    )
}

#[inline]
pub unsafe fn zscal(n: i32, a: c64, x: &mut [c64], incx: i32) {
    ffi::zscal_(
        &n,
        &a as *const _ as *const _,
        x.as_mut_ptr() as *mut _,
        &incx,
    )
}

#[inline]
pub unsafe fn zdscal(n: i32, a: f64, x: &mut [c64], incx: i32) {
    ffi::zdscal_(&n, &a, x.as_mut_ptr() as *mut _, &incx)
}

#[inline]
pub unsafe fn zcopy(n: i32, x: &[c64], incx: i32, y: &mut [c64], incy: i32) {
    ffi::zcopy_(
        &n,
        x.as_ptr() as *const _,
        &incx,
        y.as_mut_ptr() as *mut _,
        &incy,
    )
}

#[inline]
pub unsafe fn zaxpy(n: i32, alpha: c64, x: &[c64], incx: i32, y: &mut [c64], incy: i32) {
    ffi::zaxpy_(
        &n,
        &alpha as *const _ as *const _,
        x.as_ptr() as *const _,
        &incx,
        y.as_mut_ptr() as *mut _,
        &incy,
    )
}

#[inline]
pub unsafe fn zdotu(pres: &mut [c64], n: i32, x: &[c64], incx: i32, y: &[c64], incy: i32) {
    ffi::zdotu_(
        pres.as_mut_ptr() as *mut _,
        &n,
        x.as_ptr() as *const _,
        &incx,
        y.as_ptr() as *const _,
        &incy,
    )
}

#[inline]
pub unsafe fn zdotc(pres: &mut [c64], n: i32, x: &[c64], incx: i32, y: &[c64], incy: i32) {
    ffi::zdotc_(
        pres.as_mut_ptr() as *mut _,
        &n,
        x.as_ptr() as *const _,
        &incx,
        y.as_ptr() as *const _,
        &incy,
    )
}

#[inline]
pub unsafe fn dzasum(n: i32, x: &[c64], incx: i32) -> f64 {
    ffi::dzasum_(&n, x.as_ptr() as *const _, &incx)
}

#[inline]
pub unsafe fn izamax(n: i32, x: &[c64], incx: i32) -> usize {
    ffi::izamax_(&n, x.as_ptr() as *const _, &incx) as usize
}

#[inline]
pub unsafe fn sgemv(
    trans: u8,
    m: i32,
    n: i32,
    alpha: f32,
    a: &[f32],
    lda: i32,
    x: &[f32],
    incx: i32,
    beta: f32,
    y: &mut [f32],
    incy: i32,
) {
    ffi::sgemv_(
        &(trans as c_char),
        &m,
        &n,
        &alpha,
        a.as_ptr(),
        &lda,
        x.as_ptr(),
        &incx,
        &beta,
        y.as_mut_ptr(),
        &incy,
    )
}

#[inline]
pub unsafe fn sgbmv(
    trans: u8,
    m: i32,
    n: i32,
    kl: i32,
    ku: i32,
    alpha: f32,
    a: &[f32],
    lda: i32,
    x: &[f32],
    incx: i32,
    beta: f32,
    y: &mut [f32],
    incy: i32,
) {
    ffi::sgbmv_(
        &(trans as c_char),
        &m,
        &n,
        &kl,
        &ku,
        &alpha,
        a.as_ptr(),
        &lda,
        x.as_ptr(),
        &incx,
        &beta,
        y.as_mut_ptr(),
        &incy,
    )
}

#[inline]
pub unsafe fn ssymv(
    uplo: u8,
    n: i32,
    alpha: f32,
    a: &[f32],
    lda: i32,
    x: &[f32],
    incx: i32,
    beta: f32,
    y: &mut [f32],
    incy: i32,
) {
    ffi::ssymv_(
        &(uplo as c_char),
        &n,
        &alpha,
        a.as_ptr(),
        &lda,
        x.as_ptr(),
        &incx,
        &beta,
        y.as_mut_ptr(),
        &incy,
    )
}

#[inline]
pub unsafe fn ssbmv(
    uplo: u8,
    n: i32,
    k: i32,
    alpha: f32,
    a: &[f32],
    lda: i32,
    x: &[f32],
    incx: i32,
    beta: f32,
    y: &mut [f32],
    incy: i32,
) {
    ffi::ssbmv_(
        &(uplo as c_char),
        &n,
        &k,
        &alpha,
        a.as_ptr(),
        &lda,
        x.as_ptr(),
        &incx,
        &beta,
        y.as_mut_ptr(),
        &incy,
    )
}

#[inline]
pub unsafe fn sspmv(
    uplo: u8,
    n: i32,
    alpha: f32,
    ap: &[f32],
    x: &[f32],
    incx: i32,
    beta: f32,
    y: &mut [f32],
    incy: i32,
) {
    ffi::sspmv_(
        &(uplo as c_char),
        &n,
        &alpha,
        ap.as_ptr(),
        x.as_ptr(),
        &incx,
        &beta,
        y.as_mut_ptr(),
        &incy,
    )
}

#[inline]
pub unsafe fn strmv(
    uplo: u8,
    transa: u8,
    diag: u8,
    n: i32,
    a: &[f32],
    lda: i32,
    b: &mut [f32],
    incx: i32,
) {
    ffi::strmv_(
        &(uplo as c_char),
        &(transa as c_char),
        &(diag as c_char),
        &n,
        a.as_ptr(),
        &lda,
        b.as_mut_ptr(),
        &incx,
    )
}

#[inline]
pub unsafe fn stbmv(
    uplo: u8,
    trans: u8,
    diag: u8,
    n: i32,
    k: i32,
    a: &[f32],
    lda: i32,
    x: &mut [f32],
    incx: i32,
) {
    ffi::stbmv_(
        &(uplo as c_char),
        &(trans as c_char),
        &(diag as c_char),
        &n,
        &k,
        a.as_ptr(),
        &lda,
        x.as_mut_ptr(),
        &incx,
    )
}

#[inline]
pub unsafe fn stpmv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[f32], x: &mut [f32], incx: i32) {
    ffi::stpmv_(
        &(uplo as c_char),
        &(trans as c_char),
        &(diag as c_char),
        &n,
        ap.as_ptr(),
        x.as_mut_ptr(),
        &incx,
    )
}

#[inline]
pub unsafe fn strsv(
    uplo: u8,
    trans: u8,
    diag: u8,
    n: i32,
    a: &[f32],
    lda: i32,
    x: &mut [f32],
    incx: i32,
) {
    ffi::strsv_(
        &(uplo as c_char),
        &(trans as c_char),
        &(diag as c_char),
        &n,
        a.as_ptr(),
        &lda,
        x.as_mut_ptr(),
        &incx,
    )
}

#[inline]
pub unsafe fn stbsv(
    uplo: u8,
    trans: u8,
    diag: u8,
    n: i32,
    k: i32,
    a: &[f32],
    lda: i32,
    x: &mut [f32],
    incx: i32,
) {
    ffi::stbsv_(
        &(uplo as c_char),
        &(trans as c_char),
        &(diag as c_char),
        &n,
        &k,
        a.as_ptr(),
        &lda,
        x.as_mut_ptr(),
        &incx,
    )
}

#[inline]
pub unsafe fn stpsv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[f32], x: &mut [f32], incx: i32) {
    ffi::stpsv_(
        &(uplo as c_char),
        &(trans as c_char),
        &(diag as c_char),
        &n,
        ap.as_ptr(),
        x.as_mut_ptr(),
        &incx,
    )
}

#[inline]
pub unsafe fn sger(
    m: i32,
    n: i32,
    alpha: f32,
    x: &[f32],
    incx: i32,
    y: &[f32],
    incy: i32,
    a: &mut [f32],
    lda: i32,
) {
    ffi::sger_(
        &m,
        &n,
        &alpha,
        x.as_ptr(),
        &incx,
        y.as_ptr(),
        &incy,
        a.as_mut_ptr(),
        &lda,
    )
}

#[inline]
pub unsafe fn ssyr(uplo: u8, n: i32, alpha: f32, x: &[f32], incx: i32, a: &mut [f32], lda: i32) {
    ffi::ssyr_(
        &(uplo as c_char),
        &n,
        &alpha,
        x.as_ptr(),
        &incx,
        a.as_mut_ptr(),
        &lda,
    )
}

#[inline]
pub unsafe fn sspr(uplo: u8, n: i32, alpha: f32, x: &[f32], incx: i32, ap: &mut [f32]) {
    ffi::sspr_(
        &(uplo as c_char),
        &n,
        &alpha,
        x.as_ptr(),
        &incx,
        ap.as_mut_ptr(),
    )
}

#[inline]
pub unsafe fn ssyr2(
    uplo: u8,
    n: i32,
    alpha: f32,
    x: &[f32],
    incx: i32,
    y: &[f32],
    incy: i32,
    a: &mut [f32],
    lda: i32,
) {
    ffi::ssyr2_(
        &(uplo as c_char),
        &n,
        &alpha,
        x.as_ptr(),
        &incx,
        y.as_ptr(),
        &incy,
        a.as_mut_ptr(),
        &lda,
    )
}

#[inline]
pub unsafe fn sspr2(
    uplo: u8,
    n: i32,
    alpha: f32,
    x: &[f32],
    incx: i32,
    y: &[f32],
    incy: i32,
    ap: &mut [f32],
) {
    ffi::sspr2_(
        &(uplo as c_char),
        &n,
        &alpha,
        x.as_ptr(),
        &incx,
        y.as_ptr(),
        &incy,
        ap.as_mut_ptr(),
    )
}

#[inline]
pub unsafe fn dgemv(
    trans: u8,
    m: i32,
    n: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    x: &[f64],
    incx: i32,
    beta: f64,
    y: &mut [f64],
    incy: i32,
) {
    ffi::dgemv_(
        &(trans as c_char),
        &m,
        &n,
        &alpha,
        a.as_ptr(),
        &lda,
        x.as_ptr(),
        &incx,
        &beta,
        y.as_mut_ptr(),
        &incy,
    )
}

#[inline]
pub unsafe fn dgbmv(
    trans: u8,
    m: i32,
    n: i32,
    kl: i32,
    ku: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    x: &[f64],
    incx: i32,
    beta: f64,
    y: &mut [f64],
    incy: i32,
) {
    ffi::dgbmv_(
        &(trans as c_char),
        &m,
        &n,
        &kl,
        &ku,
        &alpha,
        a.as_ptr(),
        &lda,
        x.as_ptr(),
        &incx,
        &beta,
        y.as_mut_ptr(),
        &incy,
    )
}

#[inline]
pub unsafe fn dsymv(
    uplo: u8,
    n: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    x: &[f64],
    incx: i32,
    beta: f64,
    y: &mut [f64],
    incy: i32,
) {
    ffi::dsymv_(
        &(uplo as c_char),
        &n,
        &alpha,
        a.as_ptr(),
        &lda,
        x.as_ptr(),
        &incx,
        &beta,
        y.as_mut_ptr(),
        &incy,
    )
}

#[inline]
pub unsafe fn dsbmv(
    uplo: u8,
    n: i32,
    k: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    x: &[f64],
    incx: i32,
    beta: f64,
    y: &mut [f64],
    incy: i32,
) {
    ffi::dsbmv_(
        &(uplo as c_char),
        &n,
        &k,
        &alpha,
        a.as_ptr(),
        &lda,
        x.as_ptr(),
        &incx,
        &beta,
        y.as_mut_ptr(),
        &incy,
    )
}

#[inline]
pub unsafe fn dspmv(
    uplo: u8,
    n: i32,
    alpha: f64,
    ap: &[f64],
    x: &[f64],
    incx: i32,
    beta: f64,
    y: &mut [f64],
    incy: i32,
) {
    ffi::dspmv_(
        &(uplo as c_char),
        &n,
        &alpha,
        ap.as_ptr(),
        x.as_ptr(),
        &incx,
        &beta,
        y.as_mut_ptr(),
        &incy,
    )
}

#[inline]
pub unsafe fn dtrmv(
    uplo: u8,
    transa: u8,
    diag: u8,
    n: i32,
    a: &[f64],
    lda: i32,
    b: &mut [f64],
    incx: i32,
) {
    ffi::dtrmv_(
        &(uplo as c_char),
        &(transa as c_char),
        &(diag as c_char),
        &n,
        a.as_ptr(),
        &lda,
        b.as_mut_ptr(),
        &incx,
    )
}

#[inline]
pub unsafe fn dtbmv(
    uplo: u8,
    trans: u8,
    diag: u8,
    n: i32,
    k: i32,
    a: &[f64],
    lda: i32,
    x: &mut [f64],
    incx: i32,
) {
    ffi::dtbmv_(
        &(uplo as c_char),
        &(trans as c_char),
        &(diag as c_char),
        &n,
        &k,
        a.as_ptr(),
        &lda,
        x.as_mut_ptr(),
        &incx,
    )
}

#[inline]
pub unsafe fn dtpmv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[f64], x: &mut [f64], incx: i32) {
    ffi::dtpmv_(
        &(uplo as c_char),
        &(trans as c_char),
        &(diag as c_char),
        &n,
        ap.as_ptr(),
        x.as_mut_ptr(),
        &incx,
    )
}

#[inline]
pub unsafe fn dtrsv(
    uplo: u8,
    trans: u8,
    diag: u8,
    n: i32,
    a: &[f64],
    lda: i32,
    x: &mut [f64],
    incx: i32,
) {
    ffi::dtrsv_(
        &(uplo as c_char),
        &(trans as c_char),
        &(diag as c_char),
        &n,
        a.as_ptr(),
        &lda,
        x.as_mut_ptr(),
        &incx,
    )
}

#[inline]
pub unsafe fn dtbsv(
    uplo: u8,
    trans: u8,
    diag: u8,
    n: i32,
    k: i32,
    a: &[f64],
    lda: i32,
    x: &mut [f64],
    incx: i32,
) {
    ffi::dtbsv_(
        &(uplo as c_char),
        &(trans as c_char),
        &(diag as c_char),
        &n,
        &k,
        a.as_ptr(),
        &lda,
        x.as_mut_ptr(),
        &incx,
    )
}

#[inline]
pub unsafe fn dtpsv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[f64], x: &mut [f64], incx: i32) {
    ffi::dtpsv_(
        &(uplo as c_char),
        &(trans as c_char),
        &(diag as c_char),
        &n,
        ap.as_ptr(),
        x.as_mut_ptr(),
        &incx,
    )
}

#[inline]
pub unsafe fn dger(
    m: i32,
    n: i32,
    alpha: f64,
    x: &[f64],
    incx: i32,
    y: &[f64],
    incy: i32,
    a: &mut [f64],
    lda: i32,
) {
    ffi::dger_(
        &m,
        &n,
        &alpha,
        x.as_ptr(),
        &incx,
        y.as_ptr(),
        &incy,
        a.as_mut_ptr(),
        &lda,
    )
}

#[inline]
pub unsafe fn dsyr(uplo: u8, n: i32, alpha: f64, x: &[f64], incx: i32, a: &mut [f64], lda: i32) {
    ffi::dsyr_(
        &(uplo as c_char),
        &n,
        &alpha,
        x.as_ptr(),
        &incx,
        a.as_mut_ptr(),
        &lda,
    )
}

#[inline]
pub unsafe fn dspr(uplo: u8, n: i32, alpha: f64, x: &[f64], incx: i32, ap: &mut [f64]) {
    ffi::dspr_(
        &(uplo as c_char),
        &n,
        &alpha,
        x.as_ptr(),
        &incx,
        ap.as_mut_ptr(),
    )
}

#[inline]
pub unsafe fn dsyr2(
    uplo: u8,
    n: i32,
    alpha: f64,
    x: &[f64],
    incx: i32,
    y: &[f64],
    incy: i32,
    a: &mut [f64],
    lda: i32,
) {
    ffi::dsyr2_(
        &(uplo as c_char),
        &n,
        &alpha,
        x.as_ptr(),
        &incx,
        y.as_ptr(),
        &incy,
        a.as_mut_ptr(),
        &lda,
    )
}

#[inline]
pub unsafe fn dspr2(
    uplo: u8,
    n: i32,
    alpha: f64,
    x: &[f64],
    incx: i32,
    y: &[f64],
    incy: i32,
    ap: &mut [f64],
) {
    ffi::dspr2_(
        &(uplo as c_char),
        &n,
        &alpha,
        x.as_ptr(),
        &incx,
        y.as_ptr(),
        &incy,
        ap.as_mut_ptr(),
    )
}

#[inline]
pub unsafe fn cgemv(
    trans: u8,
    m: i32,
    n: i32,
    alpha: c32,
    a: &[c32],
    lda: i32,
    x: &[c32],
    incx: i32,
    beta: c32,
    y: &mut [c32],
    incy: i32,
) {
    ffi::cgemv_(
        &(trans as c_char),
        &m,
        &n,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        x.as_ptr() as *const _,
        &incx,
        &beta as *const _ as *const _,
        y.as_mut_ptr() as *mut _,
        &incy,
    )
}

#[inline]
pub unsafe fn cgbmv(
    trans: u8,
    m: i32,
    n: i32,
    kl: i32,
    ku: i32,
    alpha: c32,
    a: &[c32],
    lda: i32,
    x: &[c32],
    incx: i32,
    beta: c32,
    y: &mut [c32],
    incy: i32,
) {
    ffi::cgbmv_(
        &(trans as c_char),
        &m,
        &n,
        &kl,
        &ku,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        x.as_ptr() as *const _,
        &incx,
        &beta as *const _ as *const _,
        y.as_mut_ptr() as *mut _,
        &incy,
    )
}

#[inline]
pub unsafe fn chemv(
    uplo: u8,
    n: i32,
    alpha: c32,
    a: &[c32],
    lda: i32,
    x: &[c32],
    incx: i32,
    beta: c32,
    y: &mut [c32],
    incy: i32,
) {
    ffi::chemv_(
        &(uplo as c_char),
        &n,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        x.as_ptr() as *const _,
        &incx,
        &beta as *const _ as *const _,
        y.as_mut_ptr() as *mut _,
        &incy,
    )
}

#[inline]
pub unsafe fn chbmv(
    uplo: u8,
    n: i32,
    k: i32,
    alpha: c32,
    a: &[c32],
    lda: i32,
    x: &[c32],
    incx: i32,
    beta: c32,
    y: &mut [c32],
    incy: i32,
) {
    ffi::chbmv_(
        &(uplo as c_char),
        &n,
        &k,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        x.as_ptr() as *const _,
        &incx,
        &beta as *const _ as *const _,
        y.as_mut_ptr() as *mut _,
        &incy,
    )
}

#[inline]
pub unsafe fn chpmv(
    uplo: u8,
    n: i32,
    alpha: c32,
    ap: &[c32],
    x: &[c32],
    incx: i32,
    beta: c32,
    y: &mut [c32],
    incy: i32,
) {
    ffi::chpmv_(
        &(uplo as c_char),
        &n,
        &alpha as *const _ as *const _,
        ap.as_ptr() as *const _,
        x.as_ptr() as *const _,
        &incx,
        &beta as *const _ as *const _,
        y.as_mut_ptr() as *mut _,
        &incy,
    )
}

#[inline]
pub unsafe fn ctrmv(
    uplo: u8,
    transa: u8,
    diag: u8,
    n: i32,
    a: &[c32],
    lda: i32,
    b: &mut [c32],
    incx: i32,
) {
    ffi::ctrmv_(
        &(uplo as c_char),
        &(transa as c_char),
        &(diag as c_char),
        &n,
        a.as_ptr() as *const _,
        &lda,
        b.as_mut_ptr() as *mut _,
        &incx,
    )
}

#[inline]
pub unsafe fn ctbmv(
    uplo: u8,
    trans: u8,
    diag: u8,
    n: i32,
    k: i32,
    a: &[c32],
    lda: i32,
    x: &mut [c32],
    incx: i32,
) {
    ffi::ctbmv_(
        &(uplo as c_char),
        &(trans as c_char),
        &(diag as c_char),
        &n,
        &k,
        a.as_ptr() as *const _,
        &lda,
        x.as_mut_ptr() as *mut _,
        &incx,
    )
}

#[inline]
pub unsafe fn ctpmv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[c32], x: &mut [c32], incx: i32) {
    ffi::ctpmv_(
        &(uplo as c_char),
        &(trans as c_char),
        &(diag as c_char),
        &n,
        ap.as_ptr() as *const _,
        x.as_mut_ptr() as *mut _,
        &incx,
    )
}

#[inline]
pub unsafe fn ctrsv(
    uplo: u8,
    trans: u8,
    diag: u8,
    n: i32,
    a: &[c32],
    lda: i32,
    x: &mut [c32],
    incx: i32,
) {
    ffi::ctrsv_(
        &(uplo as c_char),
        &(trans as c_char),
        &(diag as c_char),
        &n,
        a.as_ptr() as *const _,
        &lda,
        x.as_mut_ptr() as *mut _,
        &incx,
    )
}

#[inline]
pub unsafe fn ctbsv(
    uplo: u8,
    trans: u8,
    diag: u8,
    n: i32,
    k: i32,
    a: &[c32],
    lda: i32,
    x: &mut [c32],
    incx: i32,
) {
    ffi::ctbsv_(
        &(uplo as c_char),
        &(trans as c_char),
        &(diag as c_char),
        &n,
        &k,
        a.as_ptr() as *const _,
        &lda,
        x.as_mut_ptr() as *mut _,
        &incx,
    )
}

#[inline]
pub unsafe fn ctpsv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[c32], x: &mut [c32], incx: i32) {
    ffi::ctpsv_(
        &(uplo as c_char),
        &(trans as c_char),
        &(diag as c_char),
        &n,
        ap.as_ptr() as *const _,
        x.as_mut_ptr() as *mut _,
        &incx,
    )
}

#[inline]
pub unsafe fn cgeru(
    m: i32,
    n: i32,
    alpha: c32,
    x: &[c32],
    incx: i32,
    y: &[c32],
    incy: i32,
    a: &mut [c32],
    lda: i32,
) {
    ffi::cgeru_(
        &m,
        &n,
        &alpha as *const _ as *const _,
        x.as_ptr() as *const _,
        &incx,
        y.as_ptr() as *const _,
        &incy,
        a.as_mut_ptr() as *mut _,
        &lda,
    )
}

#[inline]
pub unsafe fn cgerc(
    m: i32,
    n: i32,
    alpha: c32,
    x: &[c32],
    incx: i32,
    y: &[c32],
    incy: i32,
    a: &mut [c32],
    lda: i32,
) {
    ffi::cgerc_(
        &m,
        &n,
        &alpha as *const _ as *const _,
        x.as_ptr() as *const _,
        &incx,
        y.as_ptr() as *const _,
        &incy,
        a.as_mut_ptr() as *mut _,
        &lda,
    )
}

#[inline]
pub unsafe fn cher(uplo: u8, n: i32, alpha: f32, x: &[c32], incx: i32, a: &mut [c32], lda: i32) {
    ffi::cher_(
        &(uplo as c_char),
        &n,
        &alpha,
        x.as_ptr() as *const _,
        &incx,
        a.as_mut_ptr() as *mut _,
        &lda,
    )
}

#[inline]
pub unsafe fn chpr(uplo: u8, n: i32, alpha: f32, x: &[c32], incx: i32, ap: &mut [c32]) {
    ffi::chpr_(
        &(uplo as c_char),
        &n,
        &alpha,
        x.as_ptr() as *const _,
        &incx,
        ap.as_mut_ptr() as *mut _,
    )
}

#[inline]
pub unsafe fn chpr2(
    uplo: u8,
    n: i32,
    alpha: c32,
    x: &[c32],
    incx: i32,
    y: &[c32],
    incy: i32,
    ap: &mut [c32],
) {
    ffi::chpr2_(
        &(uplo as c_char),
        &n,
        &alpha as *const _ as *const _,
        x.as_ptr() as *const _,
        &incx,
        y.as_ptr() as *const _,
        &incy,
        ap.as_mut_ptr() as *mut _,
    )
}

#[inline]
pub unsafe fn cher2(
    uplo: u8,
    n: i32,
    alpha: c32,
    x: &[c32],
    incx: i32,
    y: &[c32],
    incy: i32,
    a: &mut [c32],
    lda: i32,
) {
    ffi::cher2_(
        &(uplo as c_char),
        &n,
        &alpha as *const _ as *const _,
        x.as_ptr() as *const _,
        &incx,
        y.as_ptr() as *const _,
        &incy,
        a.as_mut_ptr() as *mut _,
        &lda,
    )
}

#[inline]
pub unsafe fn zgemv(
    trans: u8,
    m: i32,
    n: i32,
    alpha: c64,
    a: &[c64],
    lda: i32,
    x: &[c64],
    incx: i32,
    beta: c64,
    y: &mut [c64],
    incy: i32,
) {
    ffi::zgemv_(
        &(trans as c_char),
        &m,
        &n,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        x.as_ptr() as *const _,
        &incx,
        &beta as *const _ as *const _,
        y.as_mut_ptr() as *mut _,
        &incy,
    )
}

#[inline]
pub unsafe fn zgbmv(
    trans: u8,
    m: i32,
    n: i32,
    kl: i32,
    ku: i32,
    alpha: c64,
    a: &[c64],
    lda: i32,
    x: &[c64],
    incx: i32,
    beta: c64,
    y: &mut [c64],
    incy: i32,
) {
    ffi::zgbmv_(
        &(trans as c_char),
        &m,
        &n,
        &kl,
        &ku,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        x.as_ptr() as *const _,
        &incx,
        &beta as *const _ as *const _,
        y.as_mut_ptr() as *mut _,
        &incy,
    )
}

#[inline]
pub unsafe fn zhemv(
    uplo: u8,
    n: i32,
    alpha: c64,
    a: &[c64],
    lda: i32,
    x: &[c64],
    incx: i32,
    beta: c64,
    y: &mut [c64],
    incy: i32,
) {
    ffi::zhemv_(
        &(uplo as c_char),
        &n,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        x.as_ptr() as *const _,
        &incx,
        &beta as *const _ as *const _,
        y.as_mut_ptr() as *mut _,
        &incy,
    )
}

#[inline]
pub unsafe fn zhbmv(
    uplo: u8,
    n: i32,
    k: i32,
    alpha: c64,
    a: &[c64],
    lda: i32,
    x: &[c64],
    incx: i32,
    beta: c64,
    y: &mut [c64],
    incy: i32,
) {
    ffi::zhbmv_(
        &(uplo as c_char),
        &n,
        &k,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        x.as_ptr() as *const _,
        &incx,
        &beta as *const _ as *const _,
        y.as_mut_ptr() as *mut _,
        &incy,
    )
}

#[inline]
pub unsafe fn zhpmv(
    uplo: u8,
    n: i32,
    alpha: c64,
    ap: &[c64],
    x: &[c64],
    incx: i32,
    beta: c64,
    y: &mut [c64],
    incy: i32,
) {
    ffi::zhpmv_(
        &(uplo as c_char),
        &n,
        &alpha as *const _ as *const _,
        ap.as_ptr() as *const _,
        x.as_ptr() as *const _,
        &incx,
        &beta as *const _ as *const _,
        y.as_mut_ptr() as *mut _,
        &incy,
    )
}

#[inline]
pub unsafe fn ztrmv(
    uplo: u8,
    transa: u8,
    diag: u8,
    n: i32,
    a: &[c64],
    lda: i32,
    b: &mut [c64],
    incx: i32,
) {
    ffi::ztrmv_(
        &(uplo as c_char),
        &(transa as c_char),
        &(diag as c_char),
        &n,
        a.as_ptr() as *const _,
        &lda,
        b.as_mut_ptr() as *mut _,
        &incx,
    )
}

#[inline]
pub unsafe fn ztbmv(
    uplo: u8,
    trans: u8,
    diag: u8,
    n: i32,
    k: i32,
    a: &[c64],
    lda: i32,
    x: &mut [c64],
    incx: i32,
) {
    ffi::ztbmv_(
        &(uplo as c_char),
        &(trans as c_char),
        &(diag as c_char),
        &n,
        &k,
        a.as_ptr() as *const _,
        &lda,
        x.as_mut_ptr() as *mut _,
        &incx,
    )
}

#[inline]
pub unsafe fn ztpmv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[c64], x: &mut [c64], incx: i32) {
    ffi::ztpmv_(
        &(uplo as c_char),
        &(trans as c_char),
        &(diag as c_char),
        &n,
        ap.as_ptr() as *const _,
        x.as_mut_ptr() as *mut _,
        &incx,
    )
}

#[inline]
pub unsafe fn ztrsv(
    uplo: u8,
    trans: u8,
    diag: u8,
    n: i32,
    a: &[c64],
    lda: i32,
    x: &mut [c64],
    incx: i32,
) {
    ffi::ztrsv_(
        &(uplo as c_char),
        &(trans as c_char),
        &(diag as c_char),
        &n,
        a.as_ptr() as *const _,
        &lda,
        x.as_mut_ptr() as *mut _,
        &incx,
    )
}

#[inline]
pub unsafe fn ztbsv(
    uplo: u8,
    trans: u8,
    diag: u8,
    n: i32,
    k: i32,
    a: &[c64],
    lda: i32,
    x: &mut [c64],
    incx: i32,
) {
    ffi::ztbsv_(
        &(uplo as c_char),
        &(trans as c_char),
        &(diag as c_char),
        &n,
        &k,
        a.as_ptr() as *const _,
        &lda,
        x.as_mut_ptr() as *mut _,
        &incx,
    )
}

#[inline]
pub unsafe fn ztpsv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[c64], x: &mut [c64], incx: i32) {
    ffi::ztpsv_(
        &(uplo as c_char),
        &(trans as c_char),
        &(diag as c_char),
        &n,
        ap.as_ptr() as *const _,
        x.as_mut_ptr() as *mut _,
        &incx,
    )
}

#[inline]
pub unsafe fn zgeru(
    m: i32,
    n: i32,
    alpha: c64,
    x: &[c64],
    incx: i32,
    y: &[c64],
    incy: i32,
    a: &mut [c64],
    lda: i32,
) {
    ffi::zgeru_(
        &m,
        &n,
        &alpha as *const _ as *const _,
        x.as_ptr() as *const _,
        &incx,
        y.as_ptr() as *const _,
        &incy,
        a.as_mut_ptr() as *mut _,
        &lda,
    )
}

#[inline]
pub unsafe fn zgerc(
    m: i32,
    n: i32,
    alpha: c64,
    x: &[c64],
    incx: i32,
    y: &[c64],
    incy: i32,
    a: &mut [c64],
    lda: i32,
) {
    ffi::zgerc_(
        &m,
        &n,
        &alpha as *const _ as *const _,
        x.as_ptr() as *const _,
        &incx,
        y.as_ptr() as *const _,
        &incy,
        a.as_mut_ptr() as *mut _,
        &lda,
    )
}

#[inline]
pub unsafe fn zher(uplo: u8, n: i32, alpha: f64, x: &[c64], incx: i32, a: &mut [c64], lda: i32) {
    ffi::zher_(
        &(uplo as c_char),
        &n,
        &alpha,
        x.as_ptr() as *const _,
        &incx,
        a.as_mut_ptr() as *mut _,
        &lda,
    )
}

#[inline]
pub unsafe fn zhpr(uplo: u8, n: i32, alpha: f64, x: &[c64], incx: i32, ap: &mut [c64]) {
    ffi::zhpr_(
        &(uplo as c_char),
        &n,
        &alpha,
        x.as_ptr() as *const _,
        &incx,
        ap.as_mut_ptr() as *mut _,
    )
}

#[inline]
pub unsafe fn zher2(
    uplo: u8,
    n: i32,
    alpha: c64,
    x: &[c64],
    incx: i32,
    y: &[c64],
    incy: i32,
    a: &mut [c64],
    lda: i32,
) {
    ffi::zher2_(
        &(uplo as c_char),
        &n,
        &alpha as *const _ as *const _,
        x.as_ptr() as *const _,
        &incx,
        y.as_ptr() as *const _,
        &incy,
        a.as_mut_ptr() as *mut _,
        &lda,
    )
}

#[inline]
pub unsafe fn zhpr2(
    uplo: u8,
    n: i32,
    alpha: c64,
    x: &[c64],
    incx: i32,
    y: &[c64],
    incy: i32,
    ap: &mut [c64],
) {
    ffi::zhpr2_(
        &(uplo as c_char),
        &n,
        &alpha as *const _ as *const _,
        x.as_ptr() as *const _,
        &incx,
        y.as_ptr() as *const _,
        &incy,
        ap.as_mut_ptr() as *mut _,
    )
}

#[inline]
pub unsafe fn sgemm(
    transa: u8,
    transb: u8,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: &[f32],
    lda: i32,
    b: &[f32],
    ldb: i32,
    beta: f32,
    c: &mut [f32],
    ldc: i32,
) {
    ffi::sgemm_(
        &(transa as c_char),
        &(transb as c_char),
        &m,
        &n,
        &k,
        &alpha,
        a.as_ptr(),
        &lda,
        b.as_ptr(),
        &ldb,
        &beta,
        c.as_mut_ptr(),
        &ldc,
    )
}

#[inline]
pub unsafe fn ssymm(
    side: u8,
    uplo: u8,
    m: i32,
    n: i32,
    alpha: f32,
    a: &[f32],
    lda: i32,
    b: &[f32],
    ldb: i32,
    beta: f32,
    c: &mut [f32],
    ldc: i32,
) {
    ffi::ssymm_(
        &(side as c_char),
        &(uplo as c_char),
        &m,
        &n,
        &alpha,
        a.as_ptr(),
        &lda,
        b.as_ptr(),
        &ldb,
        &beta,
        c.as_mut_ptr(),
        &ldc,
    )
}

#[inline]
pub unsafe fn ssyrk(
    uplo: u8,
    trans: u8,
    n: i32,
    k: i32,
    alpha: f32,
    a: &[f32],
    lda: i32,
    beta: f32,
    c: &mut [f32],
    ldc: i32,
) {
    ffi::ssyrk_(
        &(uplo as c_char),
        &(trans as c_char),
        &n,
        &k,
        &alpha,
        a.as_ptr(),
        &lda,
        &beta,
        c.as_mut_ptr(),
        &ldc,
    )
}

#[inline]
pub unsafe fn ssyr2k(
    uplo: u8,
    trans: u8,
    n: i32,
    k: i32,
    alpha: f32,
    a: &[f32],
    lda: i32,
    b: &[f32],
    ldb: i32,
    beta: f32,
    c: &mut [f32],
    ldc: i32,
) {
    ffi::ssyr2k_(
        &(uplo as c_char),
        &(trans as c_char),
        &n,
        &k,
        &alpha,
        a.as_ptr(),
        &lda,
        b.as_ptr(),
        &ldb,
        &beta,
        c.as_mut_ptr(),
        &ldc,
    )
}

#[inline]
pub unsafe fn strmm(
    side: u8,
    uplo: u8,
    transa: u8,
    diag: u8,
    m: i32,
    n: i32,
    alpha: f32,
    a: &[f32],
    lda: i32,
    b: &mut [f32],
    ldb: i32,
) {
    ffi::strmm_(
        &(side as c_char),
        &(uplo as c_char),
        &(transa as c_char),
        &(diag as c_char),
        &m,
        &n,
        &alpha,
        a.as_ptr(),
        &lda,
        b.as_mut_ptr(),
        &ldb,
    )
}

#[inline]
pub unsafe fn strsm(
    side: u8,
    uplo: u8,
    transa: u8,
    diag: u8,
    m: i32,
    n: i32,
    alpha: f32,
    a: &[f32],
    lda: i32,
    b: &mut [f32],
    ldb: i32,
) {
    ffi::strsm_(
        &(side as c_char),
        &(uplo as c_char),
        &(transa as c_char),
        &(diag as c_char),
        &m,
        &n,
        &alpha,
        a.as_ptr(),
        &lda,
        b.as_mut_ptr(),
        &ldb,
    )
}

#[inline]
pub unsafe fn dgemm(
    transa: u8,
    transb: u8,
    m: i32,
    n: i32,
    k: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    b: &[f64],
    ldb: i32,
    beta: f64,
    c: &mut [f64],
    ldc: i32,
) {
    ffi::dgemm_(
        &(transa as c_char),
        &(transb as c_char),
        &m,
        &n,
        &k,
        &alpha,
        a.as_ptr(),
        &lda,
        b.as_ptr(),
        &ldb,
        &beta,
        c.as_mut_ptr(),
        &ldc,
    )
}

#[inline]
pub unsafe fn dsymm(
    side: u8,
    uplo: u8,
    m: i32,
    n: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    b: &[f64],
    ldb: i32,
    beta: f64,
    c: &mut [f64],
    ldc: i32,
) {
    ffi::dsymm_(
        &(side as c_char),
        &(uplo as c_char),
        &m,
        &n,
        &alpha,
        a.as_ptr(),
        &lda,
        b.as_ptr(),
        &ldb,
        &beta,
        c.as_mut_ptr(),
        &ldc,
    )
}

#[inline]
pub unsafe fn dsyrk(
    uplo: u8,
    trans: u8,
    n: i32,
    k: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    beta: f64,
    c: &mut [f64],
    ldc: i32,
) {
    ffi::dsyrk_(
        &(uplo as c_char),
        &(trans as c_char),
        &n,
        &k,
        &alpha,
        a.as_ptr(),
        &lda,
        &beta,
        c.as_mut_ptr(),
        &ldc,
    )
}

#[inline]
pub unsafe fn dsyr2k(
    uplo: u8,
    trans: u8,
    n: i32,
    k: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    b: &[f64],
    ldb: i32,
    beta: f64,
    c: &mut [f64],
    ldc: i32,
) {
    ffi::dsyr2k_(
        &(uplo as c_char),
        &(trans as c_char),
        &n,
        &k,
        &alpha,
        a.as_ptr(),
        &lda,
        b.as_ptr(),
        &ldb,
        &beta,
        c.as_mut_ptr(),
        &ldc,
    )
}

#[inline]
pub unsafe fn dtrmm(
    side: u8,
    uplo: u8,
    transa: u8,
    diag: u8,
    m: i32,
    n: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    b: &mut [f64],
    ldb: i32,
) {
    ffi::dtrmm_(
        &(side as c_char),
        &(uplo as c_char),
        &(transa as c_char),
        &(diag as c_char),
        &m,
        &n,
        &alpha,
        a.as_ptr(),
        &lda,
        b.as_mut_ptr(),
        &ldb,
    )
}

#[inline]
pub unsafe fn dtrsm(
    side: u8,
    uplo: u8,
    transa: u8,
    diag: u8,
    m: i32,
    n: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    b: &mut [f64],
    ldb: i32,
) {
    ffi::dtrsm_(
        &(side as c_char),
        &(uplo as c_char),
        &(transa as c_char),
        &(diag as c_char),
        &m,
        &n,
        &alpha,
        a.as_ptr(),
        &lda,
        b.as_mut_ptr(),
        &ldb,
    )
}

#[inline]
pub unsafe fn cgemm(
    transa: u8,
    transb: u8,
    m: i32,
    n: i32,
    k: i32,
    alpha: c32,
    a: &[c32],
    lda: i32,
    b: &[c32],
    ldb: i32,
    beta: c32,
    c: &mut [c32],
    ldc: i32,
) {
    ffi::cgemm_(
        &(transa as c_char),
        &(transb as c_char),
        &m,
        &n,
        &k,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        b.as_ptr() as *const _,
        &ldb,
        &beta as *const _ as *const _,
        c.as_mut_ptr() as *mut _,
        &ldc,
    )
}

#[inline]
pub unsafe fn csymm(
    side: u8,
    uplo: u8,
    m: i32,
    n: i32,
    alpha: c32,
    a: &[c32],
    lda: i32,
    b: &[c32],
    ldb: i32,
    beta: c32,
    c: &mut [c32],
    ldc: i32,
) {
    ffi::csymm_(
        &(side as c_char),
        &(uplo as c_char),
        &m,
        &n,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        b.as_ptr() as *const _,
        &ldb,
        &beta as *const _ as *const _,
        c.as_mut_ptr() as *mut _,
        &ldc,
    )
}

#[inline]
pub unsafe fn chemm(
    side: u8,
    uplo: u8,
    m: i32,
    n: i32,
    alpha: c32,
    a: &[c32],
    lda: i32,
    b: &[c32],
    ldb: i32,
    beta: c32,
    c: &mut [c32],
    ldc: i32,
) {
    ffi::chemm_(
        &(side as c_char),
        &(uplo as c_char),
        &m,
        &n,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        b.as_ptr() as *const _,
        &ldb,
        &beta as *const _ as *const _,
        c.as_mut_ptr() as *mut _,
        &ldc,
    )
}

#[inline]
pub unsafe fn csyrk(
    uplo: u8,
    trans: u8,
    n: i32,
    k: i32,
    alpha: c32,
    a: &[c32],
    lda: i32,
    beta: c32,
    c: &mut [c32],
    ldc: i32,
) {
    ffi::csyrk_(
        &(uplo as c_char),
        &(trans as c_char),
        &n,
        &k,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        &beta as *const _ as *const _,
        c.as_mut_ptr() as *mut _,
        &ldc,
    )
}

#[inline]
pub unsafe fn cherk(
    uplo: u8,
    trans: u8,
    n: i32,
    k: i32,
    alpha: f32,
    a: &[c32],
    lda: i32,
    beta: f32,
    c: &mut [c32],
    ldc: i32,
) {
    ffi::cherk_(
        &(uplo as c_char),
        &(trans as c_char),
        &n,
        &k,
        &alpha,
        a.as_ptr() as *const _,
        &lda,
        &beta,
        c.as_mut_ptr() as *mut _,
        &ldc,
    )
}

#[inline]
pub unsafe fn csyr2k(
    uplo: u8,
    trans: u8,
    n: i32,
    k: i32,
    alpha: c32,
    a: &[c32],
    lda: i32,
    b: &[c32],
    ldb: i32,
    beta: c32,
    c: &mut [c32],
    ldc: i32,
) {
    ffi::csyr2k_(
        &(uplo as c_char),
        &(trans as c_char),
        &n,
        &k,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        b.as_ptr() as *const _,
        &ldb,
        &beta as *const _ as *const _,
        c.as_mut_ptr() as *mut _,
        &ldc,
    )
}

#[inline]
pub unsafe fn cher2k(
    uplo: u8,
    trans: u8,
    n: i32,
    k: i32,
    alpha: c32,
    a: &[c32],
    lda: i32,
    b: &[c32],
    ldb: i32,
    beta: f32,
    c: &mut [c32],
    ldc: i32,
) {
    ffi::cher2k_(
        &(uplo as c_char),
        &(trans as c_char),
        &n,
        &k,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        b.as_ptr() as *const _,
        &ldb,
        &beta,
        c.as_mut_ptr() as *mut _,
        &ldc,
    )
}

#[inline]
pub unsafe fn ctrmm(
    side: u8,
    uplo: u8,
    transa: u8,
    diag: u8,
    m: i32,
    n: i32,
    alpha: c32,
    a: &[c32],
    lda: i32,
    b: &mut [c32],
    ldb: i32,
) {
    ffi::ctrmm_(
        &(side as c_char),
        &(uplo as c_char),
        &(transa as c_char),
        &(diag as c_char),
        &m,
        &n,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        b.as_mut_ptr() as *mut _,
        &ldb,
    )
}

#[inline]
pub unsafe fn ctrsm(
    side: u8,
    uplo: u8,
    transa: u8,
    diag: u8,
    m: i32,
    n: i32,
    alpha: c32,
    a: &[c32],
    lda: i32,
    b: &mut [c32],
    ldb: i32,
) {
    ffi::ctrsm_(
        &(side as c_char),
        &(uplo as c_char),
        &(transa as c_char),
        &(diag as c_char),
        &m,
        &n,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        b.as_mut_ptr() as *mut _,
        &ldb,
    )
}

#[inline]
pub unsafe fn zgemm(
    transa: u8,
    transb: u8,
    m: i32,
    n: i32,
    k: i32,
    alpha: c64,
    a: &[c64],
    lda: i32,
    b: &[c64],
    ldb: i32,
    beta: c64,
    c: &mut [c64],
    ldc: i32,
) {
    ffi::zgemm_(
        &(transa as c_char),
        &(transb as c_char),
        &m,
        &n,
        &k,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        b.as_ptr() as *const _,
        &ldb,
        &beta as *const _ as *const _,
        c.as_mut_ptr() as *mut _,
        &ldc,
    )
}

#[inline]
pub unsafe fn zsymm(
    side: u8,
    uplo: u8,
    m: i32,
    n: i32,
    alpha: c64,
    a: &[c64],
    lda: i32,
    b: &[c64],
    ldb: i32,
    beta: c64,
    c: &mut [c64],
    ldc: i32,
) {
    ffi::zsymm_(
        &(side as c_char),
        &(uplo as c_char),
        &m,
        &n,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        b.as_ptr() as *const _,
        &ldb,
        &beta as *const _ as *const _,
        c.as_mut_ptr() as *mut _,
        &ldc,
    )
}

#[inline]
pub unsafe fn zhemm(
    side: u8,
    uplo: u8,
    m: i32,
    n: i32,
    alpha: c64,
    a: &[c64],
    lda: i32,
    b: &[c64],
    ldb: i32,
    beta: c64,
    c: &mut [c64],
    ldc: i32,
) {
    ffi::zhemm_(
        &(side as c_char),
        &(uplo as c_char),
        &m,
        &n,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        b.as_ptr() as *const _,
        &ldb,
        &beta as *const _ as *const _,
        c.as_mut_ptr() as *mut _,
        &ldc,
    )
}

#[inline]
pub unsafe fn zsyrk(
    uplo: u8,
    trans: u8,
    n: i32,
    k: i32,
    alpha: c64,
    a: &[c64],
    lda: i32,
    beta: c64,
    c: &mut [c64],
    ldc: i32,
) {
    ffi::zsyrk_(
        &(uplo as c_char),
        &(trans as c_char),
        &n,
        &k,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        &beta as *const _ as *const _,
        c.as_mut_ptr() as *mut _,
        &ldc,
    )
}

#[inline]
pub unsafe fn zherk(
    uplo: u8,
    trans: u8,
    n: i32,
    k: i32,
    alpha: f64,
    a: &[c64],
    lda: i32,
    beta: f64,
    c: &mut [c64],
    ldc: i32,
) {
    ffi::zherk_(
        &(uplo as c_char),
        &(trans as c_char),
        &n,
        &k,
        &alpha,
        a.as_ptr() as *const _,
        &lda,
        &beta,
        c.as_mut_ptr() as *mut _,
        &ldc,
    )
}

#[inline]
pub unsafe fn zsyr2k(
    uplo: u8,
    trans: u8,
    n: i32,
    k: i32,
    alpha: c64,
    a: &[c64],
    lda: i32,
    b: &[c64],
    ldb: i32,
    beta: c64,
    c: &mut [c64],
    ldc: i32,
) {
    ffi::zsyr2k_(
        &(uplo as c_char),
        &(trans as c_char),
        &n,
        &k,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        b.as_ptr() as *const _,
        &ldb,
        &beta as *const _ as *const _,
        c.as_mut_ptr() as *mut _,
        &ldc,
    )
}

#[inline]
pub unsafe fn zher2k(
    uplo: u8,
    trans: u8,
    n: i32,
    k: i32,
    alpha: c64,
    a: &[c64],
    lda: i32,
    b: &[c64],
    ldb: i32,
    beta: f64,
    c: &mut [c64],
    ldc: i32,
) {
    ffi::zher2k_(
        &(uplo as c_char),
        &(trans as c_char),
        &n,
        &k,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        b.as_ptr() as *const _,
        &ldb,
        &beta,
        c.as_mut_ptr() as *mut _,
        &ldc,
    )
}

#[inline]
pub unsafe fn ztrmm(
    side: u8,
    uplo: u8,
    transa: u8,
    diag: u8,
    m: i32,
    n: i32,
    alpha: c64,
    a: &[c64],
    lda: i32,
    b: &mut [c64],
    ldb: i32,
) {
    ffi::ztrmm_(
        &(side as c_char),
        &(uplo as c_char),
        &(transa as c_char),
        &(diag as c_char),
        &m,
        &n,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        b.as_mut_ptr() as *mut _,
        &ldb,
    )
}

#[inline]
pub unsafe fn ztrsm(
    side: u8,
    uplo: u8,
    transa: u8,
    diag: u8,
    m: i32,
    n: i32,
    alpha: c64,
    a: &[c64],
    lda: i32,
    b: &mut [c64],
    ldb: i32,
) {
    ffi::ztrsm_(
        &(side as c_char),
        &(uplo as c_char),
        &(transa as c_char),
        &(diag as c_char),
        &m,
        &n,
        &alpha as *const _ as *const _,
        a.as_ptr() as *const _,
        &lda,
        b.as_mut_ptr() as *mut _,
        &ldb,
    )
}
