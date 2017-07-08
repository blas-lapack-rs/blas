//! The Fortran interface.
//!
//! ## Example
//!
//! ```
//! use blas::fortran::*;
//!
//! let (m, n, k) = (2, 4, 3);
//! let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
//! let b = vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0];
//! let mut c = vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0];
//!
//! dgemm(b'N', b'N', m, n, k, 1.0, &a, m, &b, k, 1.0, &mut c, m);
//!
//! assert_eq!(&c, &vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0]);
//! ```

use blas_sys::fortran as ffi;
use libc::c_char;

use {c32, c64};

#[inline]
pub fn srotg(a: &mut f32, b: &mut f32, c: &mut f32, s: &mut f32) {
    unsafe { ffi::srotg_(a, b, c, s) }
}

#[inline]
pub fn srotmg(d1: &mut f32, d2: &mut f32, x1: &mut f32, y1: f32, param: &mut [f32]) {
    unsafe { ffi::srotmg_(d1, d2, x1, &y1, param.as_mut_ptr()) }
}

#[inline]
pub fn srot(n: i32, x: &mut [f32], incx: i32, y: &mut [f32], incy: i32, c: f32, s: f32) {
    unsafe { ffi::srot_(&n, x.as_mut_ptr(), &incx, y.as_mut_ptr(), &incy, &c, &s) }
}

#[inline]
pub fn srotm(n: i32, x: &mut [f32], incx: i32, y: &mut [f32], incy: i32, param: &[f32]) {
    unsafe {
        ffi::srotm_(
            &n,
            x.as_mut_ptr(),
            &incx,
            y.as_mut_ptr(),
            &incy,
            param.as_ptr(),
        )
    }
}

#[inline]
pub fn sswap(n: i32, x: &mut [f32], incx: i32, y: &mut [f32], incy: i32) {
    unsafe { ffi::sswap_(&n, x.as_mut_ptr(), &incx, y.as_mut_ptr(), &incy) }
}

#[inline]
pub fn sscal(n: i32, a: f32, x: &mut [f32], incx: i32) {
    unsafe { ffi::sscal_(&n, &a, x.as_mut_ptr(), &incx) }
}

#[inline]
pub fn scopy(n: i32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
    unsafe { ffi::scopy_(&n, x.as_ptr(), &incx, y.as_mut_ptr(), &incy) }
}

#[inline]
pub fn saxpy(n: i32, alpha: f32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
    unsafe { ffi::saxpy_(&n, &alpha, x.as_ptr(), &incx, y.as_mut_ptr(), &incy) }
}

#[inline]
pub fn sdot(n: i32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
    unsafe { ffi::sdot_(&n, x.as_ptr(), &incx, y.as_ptr(), &incy) }
}

#[inline]
pub fn sdsdot(n: i32, sb: &[f32], x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
    unsafe { ffi::sdsdot_(&n, sb.as_ptr(), x.as_ptr(), &incx, y.as_ptr(), &incy) }
}

#[inline]
pub fn snrm2(n: i32, x: &[f32], incx: i32) -> f32 {
    unsafe { ffi::snrm2_(&n, x.as_ptr(), &incx) }
}

#[inline]
pub fn scnrm2(n: i32, x: &[c32], incx: i32) -> f32 {
    unsafe { ffi::scnrm2_(&n, x.as_ptr() as *const _, &incx) }
}

#[inline]
pub fn sasum(n: i32, x: &[f32], incx: i32) -> f32 {
    unsafe { ffi::sasum_(&n, x.as_ptr(), &incx) }
}

#[inline]
pub fn isamax(n: i32, x: &[f32], incx: i32) -> usize {
    unsafe { ffi::isamax_(&n, x.as_ptr(), &incx) as usize }
}

#[inline]
pub fn drotg(a: &mut f64, b: &mut f64, c: &mut f64, s: &mut f64) {
    unsafe { ffi::drotg_(a, b, c, s) }
}

#[inline]
pub fn drotmg(d1: &mut f64, d2: &mut f64, x1: &mut f64, y1: f64, param: &mut [f64]) {
    unsafe { ffi::drotmg_(d1, d2, x1, &y1, param.as_mut_ptr()) }
}

#[inline]
pub fn drot(n: i32, x: &mut [f64], incx: i32, y: &mut [f64], incy: i32, c: f64, s: f64) {
    unsafe { ffi::drot_(&n, x.as_mut_ptr(), &incx, y.as_mut_ptr(), &incy, &c, &s) }
}

#[inline]
pub fn drotm(n: i32, x: &mut [f64], incx: i32, y: &mut [f64], incy: i32, param: &[f64]) {
    unsafe {
        ffi::drotm_(
            &n,
            x.as_mut_ptr(),
            &incx,
            y.as_mut_ptr(),
            &incy,
            param.as_ptr(),
        )
    }
}

#[inline]
pub fn dswap(n: i32, x: &mut [f64], incx: i32, y: &mut [f64], incy: i32) {
    unsafe { ffi::dswap_(&n, x.as_mut_ptr(), &incx, y.as_mut_ptr(), &incy) }
}

#[inline]
pub fn dscal(n: i32, a: f64, x: &mut [f64], incx: i32) {
    unsafe { ffi::dscal_(&n, &a, x.as_mut_ptr(), &incx) }
}

#[inline]
pub fn dcopy(n: i32, x: &[f64], incx: i32, y: &mut [f64], incy: i32) {
    unsafe { ffi::dcopy_(&n, x.as_ptr(), &incx, y.as_mut_ptr(), &incy) }
}

#[inline]
pub fn daxpy(n: i32, alpha: f64, x: &[f64], incx: i32, y: &mut [f64], incy: i32) {
    unsafe { ffi::daxpy_(&n, &alpha, x.as_ptr(), &incx, y.as_mut_ptr(), &incy) }
}

#[inline]
pub fn ddot(n: i32, x: &[f64], incx: i32, y: &[f64], incy: i32) -> f64 {
    unsafe { ffi::ddot_(&n, x.as_ptr(), &incx, y.as_ptr(), &incy) }
}

#[inline]
pub fn dsdot(n: i32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f64 {
    unsafe { ffi::dsdot_(&n, x.as_ptr(), &incx, y.as_ptr(), &incy) }
}

#[inline]
pub fn dnrm2(n: i32, x: &[f64], incx: i32) -> f64 {
    unsafe { ffi::dnrm2_(&n, x.as_ptr(), &incx) }
}

#[inline]
pub fn dznrm2(n: i32, x: &[c64], incx: i32) -> f64 {
    unsafe { ffi::dznrm2_(&n, x.as_ptr() as *const _, &incx) }
}

#[inline]
pub fn dasum(n: i32, x: &[f64], incx: i32) -> f64 {
    unsafe { ffi::dasum_(&n, x.as_ptr(), &incx) }
}

#[inline]
pub fn idamax(n: i32, x: &[f64], incx: i32) -> usize {
    unsafe { ffi::idamax_(&n, x.as_ptr(), &incx) as usize }
}

#[inline]
pub fn crotg(a: &mut c32, b: c32, c: &mut f32, s: &mut c32) {
    unsafe {
        ffi::crotg_(
            a as *mut _ as *mut _,
            &b as *const _ as *const _,
            c,
            s as *mut _ as *mut _,
        )
    }
}

#[inline]
pub fn csrot(n: i32, x: &mut [c32], incx: i32, y: &mut [c32], incy: i32, c: f32, s: f32) {
    unsafe {
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
}

#[inline]
pub fn cswap(n: i32, x: &mut [c32], incx: i32, y: &mut [c32], incy: i32) {
    unsafe {
        ffi::cswap_(
            &n,
            x.as_mut_ptr() as *mut _,
            &incx,
            y.as_mut_ptr() as *mut _,
            &incy,
        )
    }
}

#[inline]
pub fn cscal(n: i32, a: c32, x: &mut [c32], incx: i32) {
    unsafe {
        ffi::cscal_(
            &n,
            &a as *const _ as *const _,
            x.as_mut_ptr() as *mut _,
            &incx,
        )
    }
}

#[inline]
pub fn csscal(n: i32, a: f32, x: &mut [c32], incx: i32) {
    unsafe { ffi::csscal_(&n, &a, x.as_mut_ptr() as *mut _, &incx) }
}

#[inline]
pub fn ccopy(n: i32, x: &[c32], incx: i32, y: &mut [c32], incy: i32) {
    unsafe {
        ffi::ccopy_(
            &n,
            x.as_ptr() as *const _,
            &incx,
            y.as_mut_ptr() as *mut _,
            &incy,
        )
    }
}

#[inline]
pub fn caxpy(n: i32, alpha: c32, x: &[c32], incx: i32, y: &mut [c32], incy: i32) {
    unsafe {
        ffi::caxpy_(
            &n,
            &alpha as *const _ as *const _,
            x.as_ptr() as *const _,
            &incx,
            y.as_mut_ptr() as *mut _,
            &incy,
        )
    }
}

#[inline]
pub fn cdotu(pres: &mut [c32], n: i32, x: &[c32], incx: i32, y: &[c32], incy: i32) {
    unsafe {
        ffi::cdotu_(
            pres.as_mut_ptr() as *mut _,
            &n,
            x.as_ptr() as *const _,
            &incx,
            y.as_ptr() as *const _,
            &incy,
        )
    }
}

#[inline]
pub fn cdotc(pres: &mut [c32], n: i32, x: &[c32], incx: i32, y: &[c32], incy: i32) {
    unsafe {
        ffi::cdotc_(
            pres.as_mut_ptr() as *mut _,
            &n,
            x.as_ptr() as *const _,
            &incx,
            y.as_ptr() as *const _,
            &incy,
        )
    }
}

#[inline]
pub fn scasum(n: i32, x: &[c32], incx: i32) -> f32 {
    unsafe { ffi::scasum_(&n, x.as_ptr() as *const _, &incx) }
}

#[inline]
pub fn icamax(n: i32, x: &[c32], incx: i32) -> usize {
    unsafe { ffi::icamax_(&n, x.as_ptr() as *const _, &incx) as usize }
}

#[inline]
pub fn zrotg(a: &mut c64, b: c64, c: &mut f64, s: &mut c64) {
    unsafe {
        ffi::zrotg_(
            a as *mut _ as *mut _,
            &b as *const _ as *const _,
            c,
            s as *mut _ as *mut _,
        )
    }
}

#[inline]
pub fn zdrot(n: i32, x: &mut [c64], incx: i32, y: &mut [c64], incy: i32, c: f64, s: f64) {
    unsafe {
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
}

#[inline]
pub fn zswap(n: i32, x: &mut [c64], incx: i32, y: &mut [c64], incy: i32) {
    unsafe {
        ffi::zswap_(
            &n,
            x.as_mut_ptr() as *mut _,
            &incx,
            y.as_mut_ptr() as *mut _,
            &incy,
        )
    }
}

#[inline]
pub fn zscal(n: i32, a: c64, x: &mut [c64], incx: i32) {
    unsafe {
        ffi::zscal_(
            &n,
            &a as *const _ as *const _,
            x.as_mut_ptr() as *mut _,
            &incx,
        )
    }
}

#[inline]
pub fn zdscal(n: i32, a: f64, x: &mut [c64], incx: i32) {
    unsafe { ffi::zdscal_(&n, &a, x.as_mut_ptr() as *mut _, &incx) }
}

#[inline]
pub fn zcopy(n: i32, x: &[c64], incx: i32, y: &mut [c64], incy: i32) {
    unsafe {
        ffi::zcopy_(
            &n,
            x.as_ptr() as *const _,
            &incx,
            y.as_mut_ptr() as *mut _,
            &incy,
        )
    }
}

#[inline]
pub fn zaxpy(n: i32, alpha: c64, x: &[c64], incx: i32, y: &mut [c64], incy: i32) {
    unsafe {
        ffi::zaxpy_(
            &n,
            &alpha as *const _ as *const _,
            x.as_ptr() as *const _,
            &incx,
            y.as_mut_ptr() as *mut _,
            &incy,
        )
    }
}

#[inline]
pub fn zdotu(pres: &mut [c64], n: i32, x: &[c64], incx: i32, y: &[c64], incy: i32) {
    unsafe {
        ffi::zdotu_(
            pres.as_mut_ptr() as *mut _,
            &n,
            x.as_ptr() as *const _,
            &incx,
            y.as_ptr() as *const _,
            &incy,
        )
    }
}

#[inline]
pub fn zdotc(pres: &mut [c64], n: i32, x: &[c64], incx: i32, y: &[c64], incy: i32) {
    unsafe {
        ffi::zdotc_(
            pres.as_mut_ptr() as *mut _,
            &n,
            x.as_ptr() as *const _,
            &incx,
            y.as_ptr() as *const _,
            &incy,
        )
    }
}

#[inline]
pub fn dzasum(n: i32, x: &[c64], incx: i32) -> f64 {
    unsafe { ffi::dzasum_(&n, x.as_ptr() as *const _, &incx) }
}

#[inline]
pub fn izamax(n: i32, x: &[c64], incx: i32) -> usize {
    unsafe { ffi::izamax_(&n, x.as_ptr() as *const _, &incx) as usize }
}

#[inline]
pub fn sgemv(
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
    unsafe {
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
}

#[inline]
pub fn sgbmv(
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
    unsafe {
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
}

#[inline]
pub fn ssymv(
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
    unsafe {
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
}

#[inline]
pub fn ssbmv(
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
    unsafe {
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
}

#[inline]
pub fn sspmv(
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
    unsafe {
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
}

#[inline]
pub fn strmv(
    uplo: u8,
    transa: u8,
    diag: u8,
    n: i32,
    a: &[f32],
    lda: i32,
    b: &mut [f32],
    incx: i32,
) {
    unsafe {
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
}

#[inline]
pub fn stbmv(
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
    unsafe {
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
}

#[inline]
pub fn stpmv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[f32], x: &mut [f32], incx: i32) {
    unsafe {
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
}

#[inline]
pub fn strsv(uplo: u8, trans: u8, diag: u8, n: i32, a: &[f32], lda: i32, x: &mut [f32], incx: i32) {
    unsafe {
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
}

#[inline]
pub fn stbsv(
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
    unsafe {
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
}

#[inline]
pub fn stpsv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[f32], x: &mut [f32], incx: i32) {
    unsafe {
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
}

#[inline]
pub fn sger(
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
    unsafe {
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
}

#[inline]
pub fn ssyr(uplo: u8, n: i32, alpha: f32, x: &[f32], incx: i32, a: &mut [f32], lda: i32) {
    unsafe {
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
}

#[inline]
pub fn sspr(uplo: u8, n: i32, alpha: f32, x: &[f32], incx: i32, ap: &mut [f32]) {
    unsafe {
        ffi::sspr_(
            &(uplo as c_char),
            &n,
            &alpha,
            x.as_ptr(),
            &incx,
            ap.as_mut_ptr(),
        )
    }
}

#[inline]
pub fn ssyr2(
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
    unsafe {
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
}

#[inline]
pub fn sspr2(
    uplo: u8,
    n: i32,
    alpha: f32,
    x: &[f32],
    incx: i32,
    y: &[f32],
    incy: i32,
    ap: &mut [f32],
) {
    unsafe {
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
}

#[inline]
pub fn dgemv(
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
    unsafe {
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
}

#[inline]
pub fn dgbmv(
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
    unsafe {
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
}

#[inline]
pub fn dsymv(
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
    unsafe {
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
}

#[inline]
pub fn dsbmv(
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
    unsafe {
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
}

#[inline]
pub fn dspmv(
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
    unsafe {
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
}

#[inline]
pub fn dtrmv(
    uplo: u8,
    transa: u8,
    diag: u8,
    n: i32,
    a: &[f64],
    lda: i32,
    b: &mut [f64],
    incx: i32,
) {
    unsafe {
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
}

#[inline]
pub fn dtbmv(
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
    unsafe {
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
}

#[inline]
pub fn dtpmv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[f64], x: &mut [f64], incx: i32) {
    unsafe {
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
}

#[inline]
pub fn dtrsv(uplo: u8, trans: u8, diag: u8, n: i32, a: &[f64], lda: i32, x: &mut [f64], incx: i32) {
    unsafe {
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
}

#[inline]
pub fn dtbsv(
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
    unsafe {
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
}

#[inline]
pub fn dtpsv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[f64], x: &mut [f64], incx: i32) {
    unsafe {
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
}

#[inline]
pub fn dger(
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
    unsafe {
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
}

#[inline]
pub fn dsyr(uplo: u8, n: i32, alpha: f64, x: &[f64], incx: i32, a: &mut [f64], lda: i32) {
    unsafe {
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
}

#[inline]
pub fn dspr(uplo: u8, n: i32, alpha: f64, x: &[f64], incx: i32, ap: &mut [f64]) {
    unsafe {
        ffi::dspr_(
            &(uplo as c_char),
            &n,
            &alpha,
            x.as_ptr(),
            &incx,
            ap.as_mut_ptr(),
        )
    }
}

#[inline]
pub fn dsyr2(
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
    unsafe {
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
}

#[inline]
pub fn dspr2(
    uplo: u8,
    n: i32,
    alpha: f64,
    x: &[f64],
    incx: i32,
    y: &[f64],
    incy: i32,
    ap: &mut [f64],
) {
    unsafe {
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
}

#[inline]
pub fn cgemv(
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
    unsafe {
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
}

#[inline]
pub fn cgbmv(
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
    unsafe {
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
}

#[inline]
pub fn chemv(
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
    unsafe {
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
}

#[inline]
pub fn chbmv(
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
    unsafe {
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
}

#[inline]
pub fn chpmv(
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
    unsafe {
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
}

#[inline]
pub fn ctrmv(
    uplo: u8,
    transa: u8,
    diag: u8,
    n: i32,
    a: &[c32],
    lda: i32,
    b: &mut [c32],
    incx: i32,
) {
    unsafe {
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
}

#[inline]
pub fn ctbmv(
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
    unsafe {
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
}

#[inline]
pub fn ctpmv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[c32], x: &mut [c32], incx: i32) {
    unsafe {
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
}

#[inline]
pub fn ctrsv(uplo: u8, trans: u8, diag: u8, n: i32, a: &[c32], lda: i32, x: &mut [c32], incx: i32) {
    unsafe {
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
}

#[inline]
pub fn ctbsv(
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
    unsafe {
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
}

#[inline]
pub fn ctpsv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[c32], x: &mut [c32], incx: i32) {
    unsafe {
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
}

#[inline]
pub fn cgeru(
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
    unsafe {
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
}

#[inline]
pub fn cgerc(
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
    unsafe {
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
}

#[inline]
pub fn cher(uplo: u8, n: i32, alpha: f32, x: &[c32], incx: i32, a: &mut [c32], lda: i32) {
    unsafe {
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
}

#[inline]
pub fn chpr(uplo: u8, n: i32, alpha: f32, x: &[c32], incx: i32, ap: &mut [c32]) {
    unsafe {
        ffi::chpr_(
            &(uplo as c_char),
            &n,
            &alpha,
            x.as_ptr() as *const _,
            &incx,
            ap.as_mut_ptr() as *mut _,
        )
    }
}

#[inline]
pub fn chpr2(
    uplo: u8,
    n: i32,
    alpha: c32,
    x: &[c32],
    incx: i32,
    y: &[c32],
    incy: i32,
    ap: &mut [c32],
) {
    unsafe {
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
}

#[inline]
pub fn cher2(
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
    unsafe {
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
}

#[inline]
pub fn zgemv(
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
    unsafe {
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
}

#[inline]
pub fn zgbmv(
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
    unsafe {
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
}

#[inline]
pub fn zhemv(
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
    unsafe {
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
}

#[inline]
pub fn zhbmv(
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
    unsafe {
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
}

#[inline]
pub fn zhpmv(
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
    unsafe {
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
}

#[inline]
pub fn ztrmv(
    uplo: u8,
    transa: u8,
    diag: u8,
    n: i32,
    a: &[c64],
    lda: i32,
    b: &mut [c64],
    incx: i32,
) {
    unsafe {
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
}

#[inline]
pub fn ztbmv(
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
    unsafe {
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
}

#[inline]
pub fn ztpmv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[c64], x: &mut [c64], incx: i32) {
    unsafe {
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
}

#[inline]
pub fn ztrsv(uplo: u8, trans: u8, diag: u8, n: i32, a: &[c64], lda: i32, x: &mut [c64], incx: i32) {
    unsafe {
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
}

#[inline]
pub fn ztbsv(
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
    unsafe {
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
}

#[inline]
pub fn ztpsv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[c64], x: &mut [c64], incx: i32) {
    unsafe {
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
}

#[inline]
pub fn zgeru(
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
    unsafe {
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
}

#[inline]
pub fn zgerc(
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
    unsafe {
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
}

#[inline]
pub fn zher(uplo: u8, n: i32, alpha: f64, x: &[c64], incx: i32, a: &mut [c64], lda: i32) {
    unsafe {
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
}

#[inline]
pub fn zhpr(uplo: u8, n: i32, alpha: f64, x: &[c64], incx: i32, ap: &mut [c64]) {
    unsafe {
        ffi::zhpr_(
            &(uplo as c_char),
            &n,
            &alpha,
            x.as_ptr() as *const _,
            &incx,
            ap.as_mut_ptr() as *mut _,
        )
    }
}

#[inline]
pub fn zher2(
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
    unsafe {
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
}

#[inline]
pub fn zhpr2(
    uplo: u8,
    n: i32,
    alpha: c64,
    x: &[c64],
    incx: i32,
    y: &[c64],
    incy: i32,
    ap: &mut [c64],
) {
    unsafe {
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
}

#[inline]
pub fn sgemm(
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
    unsafe {
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
}

#[inline]
pub fn ssymm(
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
    unsafe {
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
}

#[inline]
pub fn ssyrk(
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
    unsafe {
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
}

#[inline]
pub fn ssyr2k(
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
    unsafe {
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
}

#[inline]
pub fn strmm(
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
    unsafe {
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
}

#[inline]
pub fn strsm(
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
    unsafe {
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
}

#[inline]
pub fn dgemm(
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
    unsafe {
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
}

#[inline]
pub fn dsymm(
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
    unsafe {
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
}

#[inline]
pub fn dsyrk(
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
    unsafe {
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
}

#[inline]
pub fn dsyr2k(
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
    unsafe {
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
}

#[inline]
pub fn dtrmm(
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
    unsafe {
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
}

#[inline]
pub fn dtrsm(
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
    unsafe {
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
}

#[inline]
pub fn cgemm(
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
    unsafe {
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
}

#[inline]
pub fn csymm(
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
    unsafe {
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
}

#[inline]
pub fn chemm(
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
    unsafe {
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
}

#[inline]
pub fn csyrk(
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
    unsafe {
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
}

#[inline]
pub fn cherk(
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
    unsafe {
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
}

#[inline]
pub fn csyr2k(
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
    unsafe {
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
}

#[inline]
pub fn cher2k(
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
    unsafe {
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
}

#[inline]
pub fn ctrmm(
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
    unsafe {
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
}

#[inline]
pub fn ctrsm(
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
    unsafe {
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
}

#[inline]
pub fn zgemm(
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
    unsafe {
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
}

#[inline]
pub fn zsymm(
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
    unsafe {
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
}

#[inline]
pub fn zhemm(
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
    unsafe {
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
}

#[inline]
pub fn zsyrk(
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
    unsafe {
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
}

#[inline]
pub fn zherk(
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
    unsafe {
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
}

#[inline]
pub fn zsyr2k(
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
    unsafe {
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
}

#[inline]
pub fn zher2k(
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
    unsafe {
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
}

#[inline]
pub fn ztrmm(
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
    unsafe {
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
}

#[inline]
pub fn ztrsm(
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
    unsafe {
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
}
