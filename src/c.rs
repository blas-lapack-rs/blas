//! The C interface (CBLAS).
//!
//! ## Example
//!
//! ```
//! use blas::c::*;
//!
//! let (m, n, k) = (2, 4, 3);
//! let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
//! let b = vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0];
//! let mut c = vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0];
//!
//! dgemm(Layout::ColumnMajor, Transpose::None, Transpose::None,
//!       m, n, k, 1.0, &a, m, &b, k, 1.0, &mut c, m);
//!
//! assert_eq!(&c, &vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0]);
//! ```

use blas_sys::c as ffi;

use {c32, c64};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub enum Diagonal {
    Generic = 131,
    Unit = 132,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub enum Layout {
    RowMajor = 101,
    ColumnMajor = 102,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub enum Part {
    Upper = 121,
    Lower = 122,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub enum Side {
    Left = 141,
    Right = 142,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub enum Transpose {
    None = 111,
    Ordinary = 112,
    Conjugate = 113,
}

macro_rules! convert {
    ($($from:ident => $into:ident,)*) => (
        $(
            impl From<$from> for ffi::$into {
                #[inline(always)]
                fn from(value: $from) -> ffi::$into {
                    unsafe { ::std::mem::transmute(value) }
                }
            }
        )*
    );
}

convert! {
    Diagonal => CBLAS_DIAG,
    Layout => CBLAS_LAYOUT,
    Part => CBLAS_UPLO,
    Side => CBLAS_SIDE,
    Transpose => CBLAS_TRANSPOSE,
}

#[inline]
pub fn dcabs1(z: c64) -> f64 {
    unsafe {
        ffi::cblas_dcabs1(&z as *const _ as *const _)
    }
}

#[inline]
pub fn scabs1(c: c32) -> f32 {
    unsafe {
        ffi::cblas_scabs1(&c as *const _ as *const _)
    }
}

#[inline]
pub fn sdsdot(n: i32, alpha: f32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
    unsafe {
        ffi::cblas_sdsdot(n, alpha, x.as_ptr(), incx, y.as_ptr(), incy)
    }
}

#[inline]
pub fn dsdot(n: i32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f64 {
    unsafe {
        ffi::cblas_dsdot(n, x.as_ptr(), incx, y.as_ptr(), incy)
    }
}

#[inline]
pub fn sdot(n: i32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
    unsafe {
        ffi::cblas_sdot(n, x.as_ptr(), incx, y.as_ptr(), incy)
    }
}

#[inline]
pub fn ddot(n: i32, x: &[f64], incx: i32, y: &[f64], incy: i32) -> f64 {
    unsafe {
        ffi::cblas_ddot(n, x.as_ptr(), incx, y.as_ptr(), incy)
    }
}

#[inline]
pub fn cdotu_sub(n: i32, x: &[c32], incx: i32, y: &[c32], incy: i32, dotu: &mut [c32]) {
    unsafe {
        ffi::cblas_cdotu_sub(n, x.as_ptr() as *const _, incx, y.as_ptr() as *const _, incy,
                             dotu.as_mut_ptr() as *mut _)
    }
}

#[inline]
pub fn cdotc_sub(n: i32, x: &[c32], incx: i32, y: &[c32], incy: i32, dotc: &mut [c32]) {
    unsafe {
        ffi::cblas_cdotc_sub(n, x.as_ptr() as *const _, incx, y.as_ptr() as *const _, incy,
                             dotc.as_mut_ptr() as *mut _)
    }
}

#[inline]
pub fn zdotu_sub(n: i32, x: &[c64], incx: i32, y: &[c64], incy: i32, dotu: &mut [c64]) {
    unsafe {
        ffi::cblas_zdotu_sub(n, x.as_ptr() as *const _, incx, y.as_ptr() as *const _, incy,
                             dotu.as_mut_ptr() as *mut _)
    }
}

#[inline]
pub fn zdotc_sub(n: i32, x: &[c64], incx: i32, y: &[c64], incy: i32, dotc: &mut [c64]) {
    unsafe {
        ffi::cblas_zdotc_sub(n, x.as_ptr() as *const _, incx, y.as_ptr() as *const _, incy,
                             dotc.as_mut_ptr() as *mut _)
    }
}

#[inline]
pub fn snrm2(n: i32, x: &[f32], incx: i32) -> f32 {
    unsafe {
        ffi::cblas_snrm2(n, x.as_ptr(), incx)
    }
}

#[inline]
pub fn sasum(n: i32, x: &[f32], incx: i32) -> f32 {
    unsafe {
        ffi::cblas_sasum(n, x.as_ptr(), incx)
    }
}

#[inline]
pub fn dnrm2(n: i32, x: &[f64], incx: i32) -> f64 {
    unsafe {
        ffi::cblas_dnrm2(n, x.as_ptr(), incx)
    }
}

#[inline]
pub fn dasum(n: i32, x: &[f64], incx: i32) -> f64 {
    unsafe {
        ffi::cblas_dasum(n, x.as_ptr(), incx)
    }
}

#[inline]
pub fn scnrm2(n: i32, x: &[c32], incx: i32) -> f32 {
    unsafe {
        ffi::cblas_scnrm2(n, x.as_ptr() as *const _, incx)
    }
}

#[inline]
pub fn scasum(n: i32, x: &[c32], incx: i32) -> f32 {
    unsafe {
        ffi::cblas_scasum(n, x.as_ptr() as *const _, incx)
    }
}

#[inline]
pub fn dznrm2(n: i32, x: &[c64], incx: i32) -> f64 {
    unsafe {
        ffi::cblas_dznrm2(n, x.as_ptr() as *const _, incx)
    }
}

#[inline]
pub fn dzasum(n: i32, x: &[c64], incx: i32) -> f64 {
    unsafe {
        ffi::cblas_dzasum(n, x.as_ptr() as *const _, incx)
    }
}

#[inline]
pub fn isamax(n: i32, x: &[f32], incx: i32) -> i32 {
    unsafe {
        ffi::cblas_isamax(n, x.as_ptr(), incx) as i32
    }
}

#[inline]
pub fn idamax(n: i32, x: &[f64], incx: i32) -> i32 {
    unsafe {
        ffi::cblas_idamax(n, x.as_ptr(), incx) as i32
    }
}

#[inline]
pub fn icamax(n: i32, x: &[c32], incx: i32) -> i32 {
    unsafe {
        ffi::cblas_icamax(n, x.as_ptr() as *const _, incx) as i32
    }
}

#[inline]
pub fn izamax(n: i32, x: &[c64], incx: i32) -> i32 {
    unsafe {
        ffi::cblas_izamax(n, x.as_ptr() as *const _, incx) as i32
    }
}

#[inline]
pub fn sswap(n: i32, x: &mut [f32], incx: i32, y: &mut [f32], incy: i32) {
    unsafe {
        ffi::cblas_sswap(n, x.as_mut_ptr(), incx, y.as_mut_ptr(), incy)
    }
}

#[inline]
pub fn scopy(n: i32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
    unsafe {
        ffi::cblas_scopy(n, x.as_ptr(), incx, y.as_mut_ptr(), incy)
    }
}

#[inline]
pub fn saxpy(n: i32, alpha: f32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
    unsafe {
        ffi::cblas_saxpy(n, alpha, x.as_ptr(), incx, y.as_mut_ptr(), incy)
    }
}

#[inline]
pub fn dswap(n: i32, x: &mut [f64], incx: i32, y: &mut [f64], incy: i32) {
    unsafe {
        ffi::cblas_dswap(n, x.as_mut_ptr(), incx, y.as_mut_ptr(), incy)
    }
}

#[inline]
pub fn dcopy(n: i32, x: &[f64], incx: i32, y: &mut [f64], incy: i32) {
    unsafe {
        ffi::cblas_dcopy(n, x.as_ptr(), incx, y.as_mut_ptr(), incy)
    }
}

#[inline]
pub fn daxpy(n: i32, alpha: f64, x: &[f64], incx: i32, y: &mut [f64], incy: i32) {
    unsafe {
        ffi::cblas_daxpy(n, alpha, x.as_ptr(), incx, y.as_mut_ptr(), incy)
    }
}

#[inline]
pub fn cswap(n: i32, x: &mut [c32], incx: i32, y: &mut [c32], incy: i32) {
    unsafe {
        ffi::cblas_cswap(n, x.as_mut_ptr() as *mut _, incx, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
pub fn ccopy(n: i32, x: &[c32], incx: i32, y: &mut [c32], incy: i32) {
    unsafe {
        ffi::cblas_ccopy(n, x.as_ptr() as *const _, incx, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
pub fn caxpy(n: i32, alpha: &[c32], x: &[c32], incx: i32, y: &mut [c32], incy: i32) {
    unsafe {
        ffi::cblas_caxpy(n, alpha.as_ptr() as *const _, x.as_ptr() as *const _, incx,
                         y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
pub fn zswap(n: i32, x: &mut [c64], incx: i32, y: &mut [c64], incy: i32) {
    unsafe {
        ffi::cblas_zswap(n, x.as_mut_ptr() as *mut _, incx, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
pub fn zcopy(n: i32, x: &[c64], incx: i32, y: &mut [c64], incy: i32) {
    unsafe {
        ffi::cblas_zcopy(n, x.as_ptr() as *const _, incx, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
pub fn zaxpy(n: i32, alpha: &[c64], x: &[c64], incx: i32, y: &mut [c64], incy: i32) {
    unsafe {
        ffi::cblas_zaxpy(n, alpha.as_ptr() as *const _, x.as_ptr() as *const _, incx,
                         y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
pub fn srotg(a: &mut [f32], b: &mut [f32], c: &mut f32, s: &mut [f32]) {
    unsafe {
        ffi::cblas_srotg(a.as_mut_ptr(), b.as_mut_ptr(), c, s.as_mut_ptr())
    }
}

#[inline]
pub fn srotmg(d1: &mut [f32], d2: &mut [f32], b1: &mut [f32], b2: f32, p: &mut [f32]) {
    unsafe {
        ffi::cblas_srotmg(d1.as_mut_ptr(), d2.as_mut_ptr(), b1.as_mut_ptr(), b2, p.as_mut_ptr())
    }
}

#[inline]
pub fn srot(n: i32, x: &mut [f32], incx: i32, y: &mut [f32], incy: i32, c: f32, s: f32) {
    unsafe {
        ffi::cblas_srot(n, x.as_mut_ptr(), incx, y.as_mut_ptr(), incy, c, s)
    }
}

#[inline]
pub fn srotm(n: i32, x: &mut [f32], incx: i32, y: &mut [f32], incy: i32, p: &[f32]) {
    unsafe {
        ffi::cblas_srotm(n, x.as_mut_ptr(), incx, y.as_mut_ptr(), incy, p.as_ptr())
    }
}

#[inline]
pub fn drotg(a: &mut [f64], b: &mut [f64], c: &mut f64, s: &mut [f64]) {
    unsafe {
        ffi::cblas_drotg(a.as_mut_ptr(), b.as_mut_ptr(), c, s.as_mut_ptr())
    }
}

#[inline]
pub fn drotmg(d1: &mut [f64], d2: &mut [f64], b1: &mut [f64], b2: f64, p: &mut [f64]) {
    unsafe {
        ffi::cblas_drotmg(d1.as_mut_ptr(), d2.as_mut_ptr(), b1.as_mut_ptr(), b2, p.as_mut_ptr())
    }
}

#[inline]
pub fn drot(n: i32, x: &mut [f64], incx: i32, y: &mut [f64], incy: i32, c: f64, s: f64) {
    unsafe {
        ffi::cblas_drot(n, x.as_mut_ptr(), incx, y.as_mut_ptr(), incy, c, s)
    }
}

#[inline]
pub fn drotm(n: i32, x: &mut [f64], incx: i32, y: &mut [f64], incy: i32, p: &[f64]) {
    unsafe {
        ffi::cblas_drotm(n, x.as_mut_ptr(), incx, y.as_mut_ptr(), incy, p.as_ptr())
    }
}

#[inline]
pub fn sscal(n: i32, alpha: f32, x: &mut [f32], incx: i32) {
    unsafe {
        ffi::cblas_sscal(n, alpha, x.as_mut_ptr(), incx)
    }
}

#[inline]
pub fn dscal(n: i32, alpha: f64, x: &mut [f64], incx: i32) {
    unsafe {
        ffi::cblas_dscal(n, alpha, x.as_mut_ptr(), incx)
    }
}

#[inline]
pub fn cscal(n: i32, alpha: &[c32], x: &mut [c32], incx: i32) {
    unsafe {
        ffi::cblas_cscal(n, alpha.as_ptr() as *const _, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
pub fn zscal(n: i32, alpha: &[c64], x: &mut [c64], incx: i32) {
    unsafe {
        ffi::cblas_zscal(n, alpha.as_ptr() as *const _, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
pub fn csscal(n: i32, alpha: f32, x: &mut [c32], incx: i32) {
    unsafe {
        ffi::cblas_csscal(n, alpha, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
pub fn zdscal(n: i32, alpha: f64, x: &mut [c64], incx: i32) {
    unsafe {
        ffi::cblas_zdscal(n, alpha, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
pub fn sgemv(layout: Layout, transa: Transpose, m: i32, n: i32, alpha: f32, a: &[f32], lda: i32,
             x: &[f32], incx: i32, beta: f32, y: &mut [f32], incy: i32) {

    unsafe {
        ffi::cblas_sgemv(layout.into(), transa.into(), m, n, alpha, a.as_ptr(), lda, x.as_ptr(),
                         incx, beta, y.as_mut_ptr(), incy)
    }
}

#[inline]
pub fn sgbmv(layout: Layout, transa: Transpose, m: i32, n: i32, kl: i32, ku: i32, alpha: f32,
             a: &[f32], lda: i32, x: &[f32], incx: i32, beta: f32, y: &mut [f32], incy: i32) {

    unsafe {
        ffi::cblas_sgbmv(layout.into(), transa.into(), m, n, kl, ku, alpha, a.as_ptr(), lda,
                         x.as_ptr(), incx, beta, y.as_mut_ptr(), incy)
    }
}

#[inline]
pub fn strmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, a: &[f32],
             lda: i32, x: &mut [f32], incx: i32) {

    unsafe {
        ffi::cblas_strmv(layout.into(), uplo.into(), transa.into(), diag.into(), n, a.as_ptr(),
                         lda, x.as_mut_ptr(), incx)
    }
}

#[inline]
pub fn stbmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, k: i32,
             a: &[f32], lda: i32, x: &mut [f32], incx: i32) {

    unsafe {
        ffi::cblas_stbmv(layout.into(), uplo.into(), transa.into(), diag.into(), n, k, a.as_ptr(),
                         lda, x.as_mut_ptr(), incx)
    }
}

#[inline]
pub fn stpmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, ap: &[f32],
             x: &mut [f32], incx: i32) {

    unsafe {
        ffi::cblas_stpmv(layout.into(), uplo.into(), transa.into(), diag.into(), n, ap.as_ptr(),
                         x.as_mut_ptr(), incx)
    }
}

#[inline]
pub fn strsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, a: &[f32],
             lda: i32, x: &mut [f32], incx: i32) {

    unsafe {
        ffi::cblas_strsv(layout.into(), uplo.into(), transa.into(), diag.into(), n, a.as_ptr(),
                         lda, x.as_mut_ptr(), incx)
    }
}

#[inline]
pub fn stbsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, k: i32,
             a: &[f32], lda: i32, x: &mut [f32], incx: i32) {

    unsafe {
        ffi::cblas_stbsv(layout.into(), uplo.into(), transa.into(), diag.into(), n, k, a.as_ptr(),
                         lda, x.as_mut_ptr(), incx)
    }
}

#[inline]
pub fn stpsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, ap: &[f32],
             x: &mut [f32], incx: i32) {

    unsafe {
        ffi::cblas_stpsv(layout.into(), uplo.into(), transa.into(), diag.into(), n, ap.as_ptr(),
                         x.as_mut_ptr(), incx)
    }
}

#[inline]
pub fn dgemv(layout: Layout, transa: Transpose, m: i32, n: i32, alpha: f64, a: &[f64], lda: i32,
             x: &[f64], incx: i32, beta: f64, y: &mut [f64], incy: i32) {

    unsafe {
        ffi::cblas_dgemv(layout.into(), transa.into(), m, n, alpha, a.as_ptr(), lda, x.as_ptr(),
                         incx, beta, y.as_mut_ptr(), incy)
    }
}

#[inline]
pub fn dgbmv(layout: Layout, transa: Transpose, m: i32, n: i32, kl: i32, ku: i32, alpha: f64,
             a: &[f64], lda: i32, x: &[f64], incx: i32, beta: f64, y: &mut [f64], incy: i32) {

    unsafe {
        ffi::cblas_dgbmv(layout.into(), transa.into(), m, n, kl, ku, alpha, a.as_ptr(), lda,
                         x.as_ptr(), incx, beta, y.as_mut_ptr(), incy)
    }
}

#[inline]
pub fn dtrmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, a: &[f64],
             lda: i32, x: &mut [f64], incx: i32) {

    unsafe {
        ffi::cblas_dtrmv(layout.into(), uplo.into(), transa.into(), diag.into(), n, a.as_ptr(),
                         lda, x.as_mut_ptr(), incx)
    }
}

#[inline]
pub fn dtbmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, k: i32,
             a: &[f64], lda: i32, x: &mut [f64], incx: i32) {

    unsafe {
        ffi::cblas_dtbmv(layout.into(), uplo.into(), transa.into(), diag.into(), n, k, a.as_ptr(),
                         lda, x.as_mut_ptr(), incx)
    }
}

#[inline]
pub fn dtpmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, ap: &[f64],
             x: &mut [f64], incx: i32) {

    unsafe {
        ffi::cblas_dtpmv(layout.into(), uplo.into(), transa.into(), diag.into(), n, ap.as_ptr(),
                         x.as_mut_ptr(), incx)
    }
}

#[inline]
pub fn dtrsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, a: &[f64],
             lda: i32, x: &mut [f64], incx: i32) {

    unsafe {
        ffi::cblas_dtrsv(layout.into(), uplo.into(), transa.into(), diag.into(), n, a.as_ptr(),
                         lda, x.as_mut_ptr(), incx)
    }
}

#[inline]
pub fn dtbsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, k: i32,
             a: &[f64], lda: i32, x: &mut [f64], incx: i32) {

    unsafe {
        ffi::cblas_dtbsv(layout.into(), uplo.into(), transa.into(), diag.into(), n, k, a.as_ptr(),
                         lda, x.as_mut_ptr(), incx)
    }
}

#[inline]
pub fn dtpsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, ap: &[f64],
             x: &mut [f64], incx: i32) {

    unsafe {
        ffi::cblas_dtpsv(layout.into(), uplo.into(), transa.into(), diag.into(), n, ap.as_ptr(),
                         x.as_mut_ptr(), incx)
    }
}

#[inline]
pub fn cgemv(layout: Layout, transa: Transpose, m: i32, n: i32, alpha: &[c32], a: &[c32], lda: i32,
             x: &[c32], incx: i32, beta: &[c32], y: &mut [c32], incy: i32) {

    unsafe {
        ffi::cblas_cgemv(layout.into(), transa.into(), m, n, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, x.as_ptr() as *const _, incx,
                         beta.as_ptr() as *const _, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
pub fn cgbmv(layout: Layout, transa: Transpose, m: i32, n: i32, kl: i32, ku: i32, alpha: &[c32],
             a: &[c32], lda: i32, x: &[c32], incx: i32, beta: &[c32], y: &mut [c32], incy: i32) {

    unsafe {
        ffi::cblas_cgbmv(layout.into(), transa.into(), m, n, kl, ku, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, x.as_ptr() as *const _, incx,
                         beta.as_ptr() as *const _, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
pub fn ctrmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, a: &[c32],
             lda: i32, x: &mut [c32], incx: i32) {

    unsafe {
        ffi::cblas_ctrmv(layout.into(), uplo.into(), transa.into(), diag.into(), n,
                         a.as_ptr() as *const _, lda, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
pub fn ctbmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, k: i32,
             a: &[c32], lda: i32, x: &mut [c32], incx: i32) {

    unsafe {
        ffi::cblas_ctbmv(layout.into(), uplo.into(), transa.into(), diag.into(), n, k,
                         a.as_ptr() as *const _, lda, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
pub fn ctpmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, ap: &[c32],
             x: &mut [c32], incx: i32) {

    unsafe {
        ffi::cblas_ctpmv(layout.into(), uplo.into(), transa.into(), diag.into(), n,
                         ap.as_ptr() as *const _, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
pub fn ctrsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, a: &[c32],
             lda: i32, x: &mut [c32], incx: i32) {

    unsafe {
        ffi::cblas_ctrsv(layout.into(), uplo.into(), transa.into(), diag.into(), n,
                         a.as_ptr() as *const _, lda, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
pub fn ctbsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, k: i32,
             a: &[c32], lda: i32, x: &mut [c32], incx: i32) {

    unsafe {
        ffi::cblas_ctbsv(layout.into(), uplo.into(), transa.into(), diag.into(), n, k,
                         a.as_ptr() as *const _, lda, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
pub fn ctpsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, ap: &[c32],
             x: &mut [c32], incx: i32) {

    unsafe {
        ffi::cblas_ctpsv(layout.into(), uplo.into(), transa.into(), diag.into(), n,
                         ap.as_ptr() as *const _, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
pub fn zgemv(layout: Layout, transa: Transpose, m: i32, n: i32, alpha: &[c64], a: &[c64], lda: i32,
             x: &[c64], incx: i32, beta: &[c64], y: &mut [c64], incy: i32) {

    unsafe {
        ffi::cblas_zgemv(layout.into(), transa.into(), m, n, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, x.as_ptr() as *const _, incx,
                         beta.as_ptr() as *const _, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
pub fn zgbmv(layout: Layout, transa: Transpose, m: i32, n: i32, kl: i32, ku: i32, alpha: &[c64],
             a: &[c64], lda: i32, x: &[c64], incx: i32, beta: &[c64], y: &mut [c64], incy: i32) {

    unsafe {
        ffi::cblas_zgbmv(layout.into(), transa.into(), m, n, kl, ku, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, x.as_ptr() as *const _, incx,
                         beta.as_ptr() as *const _, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
pub fn ztrmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, a: &[c64],
             lda: i32, x: &mut [c64], incx: i32) {

    unsafe {
        ffi::cblas_ztrmv(layout.into(), uplo.into(), transa.into(), diag.into(), n,
                         a.as_ptr() as *const _, lda, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
pub fn ztbmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, k: i32,
             a: &[c64], lda: i32, x: &mut [c64], incx: i32) {

    unsafe {
        ffi::cblas_ztbmv(layout.into(), uplo.into(), transa.into(), diag.into(), n, k,
                         a.as_ptr() as *const _, lda, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
pub fn ztpmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, ap: &[c64],
             x: &mut [c64], incx: i32) {

    unsafe {
        ffi::cblas_ztpmv(layout.into(), uplo.into(), transa.into(), diag.into(), n,
                         ap.as_ptr() as *const _, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
pub fn ztrsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, a: &[c64],
             lda: i32, x: &mut [c64], incx: i32) {

    unsafe {
        ffi::cblas_ztrsv(layout.into(), uplo.into(), transa.into(), diag.into(), n,
                         a.as_ptr() as *const _, lda, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
pub fn ztbsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, k: i32,
             a: &[c64], lda: i32, x: &mut [c64], incx: i32) {

    unsafe {
        ffi::cblas_ztbsv(layout.into(), uplo.into(), transa.into(), diag.into(), n, k,
                         a.as_ptr() as *const _, lda, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
pub fn ztpsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, ap: &[c64],
             x: &mut [c64], incx: i32) {

    unsafe {
        ffi::cblas_ztpsv(layout.into(), uplo.into(), transa.into(), diag.into(), n,
                         ap.as_ptr() as *const _, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
pub fn ssymv(layout: Layout, uplo: Part, n: i32, alpha: f32, a: &[f32], lda: i32, x: &[f32],
             incx: i32, beta: f32, y: &mut [f32], incy: i32) {

    unsafe {
        ffi::cblas_ssymv(layout.into(), uplo.into(), n, alpha, a.as_ptr(), lda, x.as_ptr(), incx,
                         beta, y.as_mut_ptr(), incy)
    }
}

#[inline]
pub fn ssbmv(layout: Layout, uplo: Part, n: i32, k: i32, alpha: f32, a: &[f32], lda: i32,
             x: &[f32], incx: i32, beta: f32, y: &mut [f32], incy: i32) {

    unsafe {
        ffi::cblas_ssbmv(layout.into(), uplo.into(), n, k, alpha, a.as_ptr(), lda, x.as_ptr(),
                         incx, beta, y.as_mut_ptr(), incy)
    }
}

#[inline]
pub fn sspmv(layout: Layout, uplo: Part, n: i32, alpha: f32, ap: &[f32], x: &[f32], incx: i32,
             beta: f32, y: &mut [f32], incy: i32) {

    unsafe {
        ffi::cblas_sspmv(layout.into(), uplo.into(), n, alpha, ap.as_ptr(), x.as_ptr(), incx, beta,
                         y.as_mut_ptr(), incy)
    }
}

#[inline]
pub fn sger(layout: Layout, m: i32, n: i32, alpha: f32, x: &[f32], incx: i32, y: &[f32], incy: i32,
            a: &mut [f32], lda: i32) {

    unsafe {
        ffi::cblas_sger(layout.into(), m, n, alpha, x.as_ptr(), incx, y.as_ptr(), incy,
                        a.as_mut_ptr(), lda)
    }
}

#[inline]
pub fn ssyr(layout: Layout, uplo: Part, n: i32, alpha: f32, x: &[f32], incx: i32, a: &mut [f32],
            lda: i32) {

    unsafe {
        ffi::cblas_ssyr(layout.into(), uplo.into(), n, alpha, x.as_ptr(), incx, a.as_mut_ptr(),
                        lda)
    }
}

#[inline]
pub fn sspr(layout: Layout, uplo: Part, n: i32, alpha: f32, x: &[f32], incx: i32, ap: &mut [f32]) {
    unsafe {
        ffi::cblas_sspr(layout.into(), uplo.into(), n, alpha, x.as_ptr(), incx, ap.as_mut_ptr())
    }
}

#[inline]
pub fn ssyr2(layout: Layout, uplo: Part, n: i32, alpha: f32, x: &[f32], incx: i32, y: &[f32],
             incy: i32, a: &mut [f32], lda: i32) {

    unsafe {
        ffi::cblas_ssyr2(layout.into(), uplo.into(), n, alpha, x.as_ptr(), incx, y.as_ptr(), incy,
                         a.as_mut_ptr(), lda)
    }
}

#[inline]
pub fn sspr2(layout: Layout, uplo: Part, n: i32, alpha: f32, x: &[f32], incx: i32, y: &[f32],
             incy: i32, a: &mut [f32]) {

    unsafe {
        ffi::cblas_sspr2(layout.into(), uplo.into(), n, alpha, x.as_ptr(), incx, y.as_ptr(), incy,
                         a.as_mut_ptr())
    }
}

#[inline]
pub fn dsymv(layout: Layout, uplo: Part, n: i32, alpha: f64, a: &[f64], lda: i32, x: &[f64],
             incx: i32, beta: f64, y: &mut [f64], incy: i32) {

    unsafe {
        ffi::cblas_dsymv(layout.into(), uplo.into(), n, alpha, a.as_ptr(), lda, x.as_ptr(), incx,
                         beta, y.as_mut_ptr(), incy)
    }
}

#[inline]
pub fn dsbmv(layout: Layout, uplo: Part, n: i32, k: i32, alpha: f64, a: &[f64], lda: i32,
             x: &[f64], incx: i32, beta: f64, y: &mut [f64], incy: i32) {

    unsafe {
        ffi::cblas_dsbmv(layout.into(), uplo.into(), n, k, alpha, a.as_ptr(), lda, x.as_ptr(),
                         incx, beta, y.as_mut_ptr(), incy)
    }
}

#[inline]
pub fn dspmv(layout: Layout, uplo: Part, n: i32, alpha: f64, ap: &[f64], x: &[f64], incx: i32,
             beta: f64, y: &mut [f64], incy: i32) {

    unsafe {
        ffi::cblas_dspmv(layout.into(), uplo.into(), n, alpha, ap.as_ptr(), x.as_ptr(), incx, beta,
                         y.as_mut_ptr(), incy)
    }
}

#[inline]
pub fn dger(layout: Layout, m: i32, n: i32, alpha: f64, x: &[f64], incx: i32, y: &[f64], incy: i32,
            a: &mut [f64], lda: i32) {

    unsafe {
        ffi::cblas_dger(layout.into(), m, n, alpha, x.as_ptr(), incx, y.as_ptr(), incy,
                        a.as_mut_ptr(), lda)
    }
}

#[inline]
pub fn dsyr(layout: Layout, uplo: Part, n: i32, alpha: f64, x: &[f64], incx: i32, a: &mut [f64],
            lda: i32) {

    unsafe {
        ffi::cblas_dsyr(layout.into(), uplo.into(), n, alpha, x.as_ptr(), incx, a.as_mut_ptr(),
                        lda)
    }
}

#[inline]
pub fn dspr(layout: Layout, uplo: Part, n: i32, alpha: f64, x: &[f64], incx: i32, ap: &mut [f64]) {
    unsafe {
        ffi::cblas_dspr(layout.into(), uplo.into(), n, alpha, x.as_ptr(), incx, ap.as_mut_ptr())
    }
}

#[inline]
pub fn dsyr2(layout: Layout, uplo: Part, n: i32, alpha: f64, x: &[f64], incx: i32, y: &[f64],
             incy: i32, a: &mut [f64], lda: i32) {

    unsafe {
        ffi::cblas_dsyr2(layout.into(), uplo.into(), n, alpha, x.as_ptr(), incx, y.as_ptr(), incy,
                         a.as_mut_ptr(), lda)
    }
}

#[inline]
pub fn dspr2(layout: Layout, uplo: Part, n: i32, alpha: f64, x: &[f64], incx: i32, y: &[f64],
             incy: i32, a: &mut [f64]) {

    unsafe {
        ffi::cblas_dspr2(layout.into(), uplo.into(), n, alpha, x.as_ptr(), incx, y.as_ptr(), incy,
                         a.as_mut_ptr())
    }
}

#[inline]
pub fn chemv(layout: Layout, uplo: Part, n: i32, alpha: &[c32], a: &[c32], lda: i32, x: &[c32],
             incx: i32, beta: &[c32], y: &mut [c32], incy: i32) {

    unsafe {
        ffi::cblas_chemv(layout.into(), uplo.into(), n, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, x.as_ptr() as *const _, incx,
                         beta.as_ptr() as *const _, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
pub fn chbmv(layout: Layout, uplo: Part, n: i32, k: i32, alpha: &[c32], a: &[c32], lda: i32,
             x: &[c32], incx: i32, beta: &[c32], y: &mut [c32], incy: i32) {

    unsafe {
        ffi::cblas_chbmv(layout.into(), uplo.into(), n, k, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, x.as_ptr() as *const _, incx,
                         beta.as_ptr() as *const _, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
pub fn chpmv(layout: Layout, uplo: Part, n: i32, alpha: &[c32], ap: &[c32], x: &[c32], incx: i32,
             beta: &[c32], y: &mut [c32], incy: i32) {

    unsafe {
        ffi::cblas_chpmv(layout.into(), uplo.into(), n, alpha.as_ptr() as *const _,
                         ap.as_ptr() as *const _, x.as_ptr() as *const _, incx,
                         beta.as_ptr() as *const _, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
pub fn cgeru(layout: Layout, m: i32, n: i32, alpha: &[c32], x: &[c32], incx: i32, y: &[c32],
             incy: i32, a: &mut [c32], lda: i32) {

    unsafe {
        ffi::cblas_cgeru(layout.into(), m, n, alpha.as_ptr() as *const _, x.as_ptr() as *const _,
                         incx, y.as_ptr() as *const _, incy, a.as_mut_ptr() as *mut _, lda)
    }
}

#[inline]
pub fn cgerc(layout: Layout, m: i32, n: i32, alpha: &[c32], x: &[c32], incx: i32, y: &[c32],
             incy: i32, a: &mut [c32], lda: i32) {

    unsafe {
        ffi::cblas_cgerc(layout.into(), m, n, alpha.as_ptr() as *const _, x.as_ptr() as *const _,
                         incx, y.as_ptr() as *const _, incy, a.as_mut_ptr() as *mut _, lda)
    }
}

#[inline]
pub fn cher(layout: Layout, uplo: Part, n: i32, alpha: f32, x: &[c32], incx: i32, a: &mut [c32],
            lda: i32) {

    unsafe {
        ffi::cblas_cher(layout.into(), uplo.into(), n, alpha, x.as_ptr() as *const _, incx,
                        a.as_mut_ptr() as *mut _, lda)
    }
}

#[inline]
pub fn chpr(layout: Layout, uplo: Part, n: i32, alpha: f32, x: &[c32], incx: i32, a: &mut [c32]) {
    unsafe {
        ffi::cblas_chpr(layout.into(), uplo.into(), n, alpha, x.as_ptr() as *const _, incx,
                        a.as_mut_ptr() as *mut _)
    }
}

#[inline]
pub fn cher2(layout: Layout, uplo: Part, n: i32, alpha: &[c32], x: &[c32], incx: i32, y: &[c32],
             incy: i32, a: &mut [c32], lda: i32) {

    unsafe {
        ffi::cblas_cher2(layout.into(), uplo.into(), n, alpha.as_ptr() as *const _,
                         x.as_ptr() as *const _, incx, y.as_ptr() as *const _, incy,
                         a.as_mut_ptr() as *mut _, lda)
    }
}

#[inline]
pub fn chpr2(layout: Layout, uplo: Part, n: i32, alpha: &[c32], x: &[c32], incx: i32, y: &[c32],
             incy: i32, ap: &mut [c32]) {

    unsafe {
        ffi::cblas_chpr2(layout.into(), uplo.into(), n, alpha.as_ptr() as *const _,
                         x.as_ptr() as *const _, incx, y.as_ptr() as *const _, incy,
                         ap.as_mut_ptr() as *mut _)
    }
}

#[inline]
pub fn zhemv(layout: Layout, uplo: Part, n: i32, alpha: &[c64], a: &[c64], lda: i32, x: &[c64],
             incx: i32, beta: &[c64], y: &mut [c64], incy: i32) {

    unsafe {
        ffi::cblas_zhemv(layout.into(), uplo.into(), n, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, x.as_ptr() as *const _, incx,
                         beta.as_ptr() as *const _, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
pub fn zhbmv(layout: Layout, uplo: Part, n: i32, k: i32, alpha: &[c64], a: &[c64], lda: i32,
             x: &[c64], incx: i32, beta: &[c64], y: &mut [c64], incy: i32) {

    unsafe {
        ffi::cblas_zhbmv(layout.into(), uplo.into(), n, k, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, x.as_ptr() as *const _, incx,
                         beta.as_ptr() as *const _, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
pub fn zhpmv(layout: Layout, uplo: Part, n: i32, alpha: &[c64], ap: &[c64], x: &[c64], incx: i32,
             beta: &[c64], y: &mut [c64], incy: i32) {

    unsafe {
        ffi::cblas_zhpmv(layout.into(), uplo.into(), n, alpha.as_ptr() as *const _,
                         ap.as_ptr() as *const _, x.as_ptr() as *const _, incx,
                         beta.as_ptr() as *const _, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
pub fn zgeru(layout: Layout, m: i32, n: i32, alpha: &[c64], x: &[c64], incx: i32, y: &[c64],
             incy: i32, a: &mut [c64], lda: i32) {

    unsafe {
        ffi::cblas_zgeru(layout.into(), m, n, alpha.as_ptr() as *const _, x.as_ptr() as *const _,
                         incx, y.as_ptr() as *const _, incy, a.as_mut_ptr() as *mut _, lda)
    }
}

#[inline]
pub fn zgerc(layout: Layout, m: i32, n: i32, alpha: &[c64], x: &[c64], incx: i32, y: &[c64],
             incy: i32, a: &mut [c64], lda: i32) {

    unsafe {
        ffi::cblas_zgerc(layout.into(), m, n, alpha.as_ptr() as *const _, x.as_ptr() as *const _,
                         incx, y.as_ptr() as *const _, incy, a.as_mut_ptr() as *mut _, lda)
    }
}

#[inline]
pub fn zher(layout: Layout, uplo: Part, n: i32, alpha: f64, x: &[c64], incx: i32, a: &mut [c64],
            lda: i32) {

    unsafe {
        ffi::cblas_zher(layout.into(), uplo.into(), n, alpha, x.as_ptr() as *const _, incx,
                        a.as_mut_ptr() as *mut _, lda)
    }
}

#[inline]
pub fn zhpr(layout: Layout, uplo: Part, n: i32, alpha: f64, x: &[c64], incx: i32, a: &mut [c64]) {
    unsafe {
        ffi::cblas_zhpr(layout.into(), uplo.into(), n, alpha, x.as_ptr() as *const _, incx,
                        a.as_mut_ptr() as *mut _)
    }
}

#[inline]
pub fn zher2(layout: Layout, uplo: Part, n: i32, alpha: &[c64], x: &[c64], incx: i32, y: &[c64],
             incy: i32, a: &mut [c64], lda: i32) {

    unsafe {
        ffi::cblas_zher2(layout.into(), uplo.into(), n, alpha.as_ptr() as *const _,
                         x.as_ptr() as *const _, incx, y.as_ptr() as *const _, incy,
                         a.as_mut_ptr() as *mut _, lda)
    }
}

#[inline]
pub fn zhpr2(layout: Layout, uplo: Part, n: i32, alpha: &[c64], x: &[c64], incx: i32, y: &[c64],
             incy: i32, ap: &mut [c64]) {

    unsafe {
        ffi::cblas_zhpr2(layout.into(), uplo.into(), n, alpha.as_ptr() as *const _,
                         x.as_ptr() as *const _, incx, y.as_ptr() as *const _, incy,
                         ap.as_mut_ptr() as *mut _)
    }
}

#[inline]
pub fn sgemm(layout: Layout, transa: Transpose, transb: Transpose, m: i32, n: i32, k: i32,
             alpha: f32, a: &[f32], lda: i32, b: &[f32], ldb: i32, beta: f32, c: &mut [f32],
             ldc: i32) {

    unsafe {
        ffi::cblas_sgemm(layout.into(), transa.into(), transb.into(), m, n, k, alpha, a.as_ptr(),
                         lda, b.as_ptr(), ldb, beta, c.as_mut_ptr(), ldc)
    }
}

#[inline]
pub fn ssymm(layout: Layout, side: Side, uplo: Part, m: i32, n: i32, alpha: f32, a: &[f32],
             lda: i32, b: &[f32], ldb: i32, beta: f32, c: &mut [f32], ldc: i32) {

    unsafe {
        ffi::cblas_ssymm(layout.into(), side.into(), uplo.into(), m, n, alpha, a.as_ptr(), lda,
                         b.as_ptr(), ldb, beta, c.as_mut_ptr(), ldc)
    }
}

#[inline]
pub fn ssyrk(layout: Layout, uplo: Part, trans: Transpose, n: i32, k: i32, alpha: f32, a: &[f32],
             lda: i32, beta: f32, c: &mut [f32], ldc: i32) {

    unsafe {
        ffi::cblas_ssyrk(layout.into(), uplo.into(), trans.into(), n, k, alpha, a.as_ptr(), lda,
                         beta, c.as_mut_ptr(), ldc)
    }
}

#[inline]
pub fn ssyr2k(layout: Layout, uplo: Part, trans: Transpose, n: i32, k: i32, alpha: f32, a: &[f32],
              lda: i32, b: &[f32], ldb: i32, beta: f32, c: &mut [f32], ldc: i32) {

    unsafe {
        ffi::cblas_ssyr2k(layout.into(), uplo.into(), trans.into(), n, k, alpha, a.as_ptr(), lda,
                          b.as_ptr(), ldb, beta, c.as_mut_ptr(), ldc)
    }
}

#[inline]
pub fn strmm(layout: Layout, side: Side, uplo: Part, transa: Transpose, diag: Diagonal, m: i32,
             n: i32, alpha: f32, a: &[f32], lda: i32, b: &mut [f32], ldb: i32) {

    unsafe {
        ffi::cblas_strmm(layout.into(), side.into(), uplo.into(), transa.into(), diag.into(), m, n,
                         alpha, a.as_ptr(), lda, b.as_mut_ptr(), ldb)
    }
}

#[inline]
pub fn strsm(layout: Layout, side: Side, uplo: Part, transa: Transpose, diag: Diagonal, m: i32,
             n: i32, alpha: f32, a: &[f32], lda: i32, b: &mut [f32], ldb: i32) {

    unsafe {
        ffi::cblas_strsm(layout.into(), side.into(), uplo.into(), transa.into(), diag.into(), m, n,
                         alpha, a.as_ptr(), lda, b.as_mut_ptr(), ldb)
    }
}

#[inline]
pub fn dgemm(layout: Layout, transa: Transpose, transb: Transpose, m: i32, n: i32, k: i32,
             alpha: f64, a: &[f64], lda: i32, b: &[f64], ldb: i32, beta: f64, c: &mut [f64],
             ldc: i32) {

    unsafe {
        ffi::cblas_dgemm(layout.into(), transa.into(), transb.into(), m, n, k, alpha, a.as_ptr(),
                         lda, b.as_ptr(), ldb, beta, c.as_mut_ptr(), ldc)
    }
}

#[inline]
pub fn dsymm(layout: Layout, side: Side, uplo: Part, m: i32, n: i32, alpha: f64, a: &[f64],
             lda: i32, b: &[f64], ldb: i32, beta: f64, c: &mut [f64], ldc: i32) {

    unsafe {
        ffi::cblas_dsymm(layout.into(), side.into(), uplo.into(), m, n, alpha, a.as_ptr(), lda,
                         b.as_ptr(), ldb, beta, c.as_mut_ptr(), ldc)
    }
}

#[inline]
pub fn dsyrk(layout: Layout, uplo: Part, trans: Transpose, n: i32, k: i32, alpha: f64, a: &[f64],
             lda: i32, beta: f64, c: &mut [f64], ldc: i32) {

    unsafe {
        ffi::cblas_dsyrk(layout.into(), uplo.into(), trans.into(), n, k, alpha, a.as_ptr(), lda,
                         beta, c.as_mut_ptr(), ldc)
    }
}

#[inline]
pub fn dsyr2k(layout: Layout, uplo: Part, trans: Transpose, n: i32, k: i32, alpha: f64, a: &[f64],
              lda: i32, b: &[f64], ldb: i32, beta: f64, c: &mut [f64], ldc: i32) {

    unsafe {
        ffi::cblas_dsyr2k(layout.into(), uplo.into(), trans.into(), n, k, alpha, a.as_ptr(), lda,
                          b.as_ptr(), ldb, beta, c.as_mut_ptr(), ldc)
    }
}

#[inline]
pub fn dtrmm(layout: Layout, side: Side, uplo: Part, transa: Transpose, diag: Diagonal, m: i32,
             n: i32, alpha: f64, a: &[f64], lda: i32, b: &mut [f64], ldb: i32) {

    unsafe {
        ffi::cblas_dtrmm(layout.into(), side.into(), uplo.into(), transa.into(), diag.into(), m, n,
                         alpha, a.as_ptr(), lda, b.as_mut_ptr(), ldb)
    }
}

#[inline]
pub fn dtrsm(layout: Layout, side: Side, uplo: Part, transa: Transpose, diag: Diagonal, m: i32,
             n: i32, alpha: f64, a: &[f64], lda: i32, b: &mut [f64], ldb: i32) {

    unsafe {
        ffi::cblas_dtrsm(layout.into(), side.into(), uplo.into(), transa.into(), diag.into(), m, n,
                         alpha, a.as_ptr(), lda, b.as_mut_ptr(), ldb)
    }
}

#[inline]
pub fn cgemm(layout: Layout, transa: Transpose, transb: Transpose, m: i32, n: i32, k: i32,
             alpha: &[c32], a: &[c32], lda: i32, b: &[c32], ldb: i32, beta: &[c32], c: &mut [c32],
             ldc: i32) {

    unsafe {
        ffi::cblas_cgemm(layout.into(), transa.into(), transb.into(), m, n, k,
                         alpha.as_ptr() as *const _, a.as_ptr() as *const _, lda,
                         b.as_ptr() as *const _, ldb, beta.as_ptr() as *const _,
                         c.as_mut_ptr() as *mut _, ldc)
    }
}

#[inline]
pub fn csymm(layout: Layout, side: Side, uplo: Part, m: i32, n: i32, alpha: &[c32], a: &[c32],
             lda: i32, b: &[c32], ldb: i32, beta: &[c32], c: &mut [c32], ldc: i32) {

    unsafe {
        ffi::cblas_csymm(layout.into(), side.into(), uplo.into(), m, n, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, b.as_ptr() as *const _, ldb,
                         beta.as_ptr() as *const _, c.as_mut_ptr() as *mut _, ldc)
    }
}

#[inline]
pub fn csyrk(layout: Layout, uplo: Part, trans: Transpose, n: i32, k: i32, alpha: &[c32],
             a: &[c32], lda: i32, beta: &[c32], c: &mut [c32], ldc: i32) {

    unsafe {
        ffi::cblas_csyrk(layout.into(), uplo.into(), trans.into(), n, k,
                         alpha.as_ptr() as *const _, a.as_ptr() as *const _, lda,
                         beta.as_ptr() as *const _, c.as_mut_ptr() as *mut _, ldc)
    }
}

#[inline]
pub fn csyr2k(layout: Layout, uplo: Part, trans: Transpose, n: i32, k: i32, alpha: &[c32],
              a: &[c32], lda: i32, b: &[c32], ldb: i32, beta: &[c32], c: &mut [c32], ldc: i32) {

    unsafe {
        ffi::cblas_csyr2k(layout.into(), uplo.into(), trans.into(), n, k,
                          alpha.as_ptr() as *const _, a.as_ptr() as *const _, lda,
                          b.as_ptr() as *const _, ldb, beta.as_ptr() as *const _,
                          c.as_mut_ptr() as *mut _, ldc)
    }
}

#[inline]
pub fn ctrmm(layout: Layout, side: Side, uplo: Part, transa: Transpose, diag: Diagonal, m: i32,
             n: i32, alpha: &[c32], a: &[c32], lda: i32, b: &mut [c32], ldb: i32) {

    unsafe {
        ffi::cblas_ctrmm(layout.into(), side.into(), uplo.into(), transa.into(), diag.into(), m, n,
                         alpha.as_ptr() as *const _, a.as_ptr() as *const _, lda,
                         b.as_mut_ptr() as *mut _, ldb)
    }
}

#[inline]
pub fn ctrsm(layout: Layout, side: Side, uplo: Part, transa: Transpose, diag: Diagonal, m: i32,
             n: i32, alpha: &[c32], a: &[c32], lda: i32, b: &mut [c32], ldb: i32) {

    unsafe {
        ffi::cblas_ctrsm(layout.into(), side.into(), uplo.into(), transa.into(), diag.into(), m, n,
                         alpha.as_ptr() as *const _, a.as_ptr() as *const _, lda,
                         b.as_mut_ptr() as *mut _, ldb)
    }
}

#[inline]
pub fn zgemm(layout: Layout, transa: Transpose, transb: Transpose, m: i32, n: i32, k: i32,
             alpha: &[c64], a: &[c64], lda: i32, b: &[c64], ldb: i32, beta: &[c64], c: &mut [c64],
             ldc: i32) {

    unsafe {
        ffi::cblas_zgemm(layout.into(), transa.into(), transb.into(), m, n, k,
                         alpha.as_ptr() as *const _, a.as_ptr() as *const _, lda,
                         b.as_ptr() as *const _, ldb, beta.as_ptr() as *const _,
                         c.as_mut_ptr() as *mut _, ldc)
    }
}

#[inline]
pub fn zsymm(layout: Layout, side: Side, uplo: Part, m: i32, n: i32, alpha: &[c64], a: &[c64],
             lda: i32, b: &[c64], ldb: i32, beta: &[c64], c: &mut [c64], ldc: i32) {

    unsafe {
        ffi::cblas_zsymm(layout.into(), side.into(), uplo.into(), m, n, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, b.as_ptr() as *const _, ldb,
                         beta.as_ptr() as *const _, c.as_mut_ptr() as *mut _, ldc)
    }
}

#[inline]
pub fn zsyrk(layout: Layout, uplo: Part, trans: Transpose, n: i32, k: i32, alpha: &[c64],
             a: &[c64], lda: i32, beta: &[c64], c: &mut [c64], ldc: i32) {

    unsafe {
        ffi::cblas_zsyrk(layout.into(), uplo.into(), trans.into(), n, k,
                         alpha.as_ptr() as *const _, a.as_ptr() as *const _, lda,
                         beta.as_ptr() as *const _, c.as_mut_ptr() as *mut _, ldc)
    }
}

#[inline]
pub fn zsyr2k(layout: Layout, uplo: Part, trans: Transpose, n: i32, k: i32, alpha: &[c64],
              a: &[c64], lda: i32, b: &[c64], ldb: i32, beta: &[c64], c: &mut [c64], ldc: i32) {

    unsafe {
        ffi::cblas_zsyr2k(layout.into(), uplo.into(), trans.into(), n, k,
                          alpha.as_ptr() as *const _, a.as_ptr() as *const _, lda,
                          b.as_ptr() as *const _, ldb, beta.as_ptr() as *const _,
                          c.as_mut_ptr() as *mut _, ldc)
    }
}

#[inline]
pub fn ztrmm(layout: Layout, side: Side, uplo: Part, transa: Transpose, diag: Diagonal, m: i32,
             n: i32, alpha: &[c64], a: &[c64], lda: i32, b: &mut [c64], ldb: i32) {

    unsafe {
        ffi::cblas_ztrmm(layout.into(), side.into(), uplo.into(), transa.into(), diag.into(), m, n,
                         alpha.as_ptr() as *const _, a.as_ptr() as *const _, lda,
                         b.as_mut_ptr() as *mut _, ldb)
    }
}

#[inline]
pub fn ztrsm(layout: Layout, side: Side, uplo: Part, transa: Transpose, diag: Diagonal, m: i32,
             n: i32, alpha: &[c64], a: &[c64], lda: i32, b: &mut [c64], ldb: i32) {

    unsafe {
        ffi::cblas_ztrsm(layout.into(), side.into(), uplo.into(), transa.into(), diag.into(), m, n,
                         alpha.as_ptr() as *const _, a.as_ptr() as *const _, lda,
                         b.as_mut_ptr() as *mut _, ldb)
    }
}

#[inline]
pub fn chemm(layout: Layout, side: Side, uplo: Part, m: i32, n: i32, alpha: &[c32], a: &[c32],
             lda: i32, b: &[c32], ldb: i32, beta: &[c32], c: &mut [c32], ldc: i32) {

    unsafe {
        ffi::cblas_chemm(layout.into(), side.into(), uplo.into(), m, n, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, b.as_ptr() as *const _, ldb,
                         beta.as_ptr() as *const _, c.as_mut_ptr() as *mut _, ldc)
    }
}

#[inline]
pub fn cherk(layout: Layout, uplo: Part, trans: Transpose, n: i32, k: i32, alpha: f32, a: &[c32],
             lda: i32, beta: f32, c: &mut [c32], ldc: i32) {

    unsafe {
        ffi::cblas_cherk(layout.into(), uplo.into(), trans.into(), n, k, alpha,
                         a.as_ptr() as *const _, lda, beta, c.as_mut_ptr() as *mut _, ldc)
    }
}

#[inline]
pub fn cher2k(layout: Layout, uplo: Part, trans: Transpose, n: i32, k: i32, alpha: &[c32],
              a: &[c32], lda: i32, b: &[c32], ldb: i32, beta: f32, c: &mut [c32], ldc: i32) {

    unsafe {
        ffi::cblas_cher2k(layout.into(), uplo.into(), trans.into(), n, k,
                          alpha.as_ptr() as *const _, a.as_ptr() as *const _, lda,
                          b.as_ptr() as *const _, ldb, beta, c.as_mut_ptr() as *mut _, ldc)
    }
}

#[inline]
pub fn zhemm(layout: Layout, side: Side, uplo: Part, m: i32, n: i32, alpha: &[c64], a: &[c64],
             lda: i32, b: &[c64], ldb: i32, beta: &[c64], c: &mut [c64], ldc: i32) {

    unsafe {
        ffi::cblas_zhemm(layout.into(), side.into(), uplo.into(), m, n, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, b.as_ptr() as *const _, ldb,
                         beta.as_ptr() as *const _, c.as_mut_ptr() as *mut _, ldc)
    }
}

#[inline]
pub fn zherk(layout: Layout, uplo: Part, trans: Transpose, n: i32, k: i32, alpha: f64, a: &[c64],
             lda: i32, beta: f64, c: &mut [c64], ldc: i32) {

    unsafe {
        ffi::cblas_zherk(layout.into(), uplo.into(), trans.into(), n, k, alpha,
                         a.as_ptr() as *const _, lda, beta, c.as_mut_ptr() as *mut _, ldc)
    }
}

#[inline]
pub fn zher2k(layout: Layout, uplo: Part, trans: Transpose, n: i32, k: i32, alpha: &[c64],
              a: &[c64], lda: i32, b: &[c64], ldb: i32, beta: f64, c: &mut [c64], ldc: i32) {

    unsafe {
        ffi::cblas_zher2k(layout.into(), uplo.into(), trans.into(), n, k,
                          alpha.as_ptr() as *const _, a.as_ptr() as *const _, lda,
                          b.as_ptr() as *const _, ldb, beta, c.as_mut_ptr() as *mut _, ldc)
    }
}
