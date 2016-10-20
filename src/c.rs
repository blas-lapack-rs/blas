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
/// absolute value of complex number (real double-precision)
/// 
/// DCABS1 computes |Re(.)| + |Im(.)| of a double complex number
pub fn dcabs1(z: c64) -> f64 {
    unsafe {
        ffi::cblas_dcabs1(&z as *const _ as *const _)
    }
}

#[inline]
/// absolute value of complex number (real single-precision)
/// 
/// SCABS1 computes |Re(.)| + |Im(.)| of a complex number
pub fn scabs1(c: c32) -> f32 {
    unsafe {
        ffi::cblas_scabs1(&c as *const _ as *const _)
    }
}

#[inline]
/// dot product with extended precision accumulation (real single-precision)
/// 
///
pub fn sdsdot(n: i32, alpha: f32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
    unsafe {
        ffi::cblas_sdsdot(n, alpha, x.as_ptr(), incx, y.as_ptr(), incy)
    }
}

#[inline]
/// dot product with extended precision accumulation (real double-precision)
/// 
/// Compute the inner product of two vectors with extended
/// precision accumulation and result.
/// 
/// Returns D.P. dot product accumulated in D.P., for S.P. SX and SY
/// DSDOT = sum for I = 0 to N-1 of  SX(LX+I*INCX) * SY(LY+I*INCY),
/// where LX = 1 if INCX .GE. 0, else LX = 1+(1-N)*INCX, and LY is
/// defined in a similar way using INCY.
pub fn dsdot(n: i32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f64 {
    unsafe {
        ffi::cblas_dsdot(n, x.as_ptr(), incx, y.as_ptr(), incy)
    }
}

#[inline]
/// dot product (real single-precision)
/// 
/// SDOT forms the dot product of two vectors.
/// uses unrolled loops for increments equal to one.
pub fn sdot(n: i32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
    unsafe {
        ffi::cblas_sdot(n, x.as_ptr(), incx, y.as_ptr(), incy)
    }
}

#[inline]
/// dot product (real double-precision)
/// 
/// DDOT forms the dot product of two vectors.
/// uses unrolled loops for increments equal to one.
pub fn ddot(n: i32, x: &[f64], incx: i32, y: &[f64], incy: i32) -> f64 {
    unsafe {
        ffi::cblas_ddot(n, x.as_ptr(), incx, y.as_ptr(), incy)
    }
}

#[inline]
/// dotu subroutine with return value as argument (complex single-precision)
///
pub fn cdotu_sub(n: i32, x: &[c32], incx: i32, y: &[c32], incy: i32, dotu: &mut [c32]) {
    unsafe {
        ffi::cblas_cdotu_sub(n, x.as_ptr() as *const _, incx, y.as_ptr() as *const _, incy,
                             dotu.as_mut_ptr() as *mut _)
    }
}

#[inline]
/// dotc subroutine with return value as argument (complex single-precision)
///
pub fn cdotc_sub(n: i32, x: &[c32], incx: i32, y: &[c32], incy: i32, dotc: &mut [c32]) {
    unsafe {
        ffi::cblas_cdotc_sub(n, x.as_ptr() as *const _, incx, y.as_ptr() as *const _, incy,
                             dotc.as_mut_ptr() as *mut _)
    }
}

#[inline]
/// dotu subroutine with return value as argument (complex double-precision)
///
pub fn zdotu_sub(n: i32, x: &[c64], incx: i32, y: &[c64], incy: i32, dotu: &mut [c64]) {
    unsafe {
        ffi::cblas_zdotu_sub(n, x.as_ptr() as *const _, incx, y.as_ptr() as *const _, incy,
                             dotu.as_mut_ptr() as *mut _)
    }
}

#[inline]
/// dotc subroutine with return value as argument (complex double-precision)
///
pub fn zdotc_sub(n: i32, x: &[c64], incx: i32, y: &[c64], incy: i32, dotc: &mut [c64]) {
    unsafe {
        ffi::cblas_zdotc_sub(n, x.as_ptr() as *const _, incx, y.as_ptr() as *const _, incy,
                             dotc.as_mut_ptr() as *mut _)
    }
}

#[inline]
/// Euclidean norm (real single-precision)
/// 
/// SNRM2 returns the euclidean norm of a vector via the function
/// name, so that
/// 
/// ```text
///    SNRM2 := sqrt( x'*x ).
/// ```
pub fn snrm2(n: i32, x: &[f32], incx: i32) -> f32 {
    unsafe {
        ffi::cblas_snrm2(n, x.as_ptr(), incx)
    }
}

#[inline]
/// sum of absolute values (real single-precision)
/// 
/// SASUM takes the sum of the absolute values.
/// uses unrolled loops for increment equal to one.
pub fn sasum(n: i32, x: &[f32], incx: i32) -> f32 {
    unsafe {
        ffi::cblas_sasum(n, x.as_ptr(), incx)
    }
}

#[inline]
/// Euclidean norm (real double-precision)
/// 
/// DNRM2 returns the euclidean norm of a vector via the function
/// name, so that
/// 
/// ```text
///    DNRM2 := sqrt( x'*x )
/// ```
pub fn dnrm2(n: i32, x: &[f64], incx: i32) -> f64 {
    unsafe {
        ffi::cblas_dnrm2(n, x.as_ptr(), incx)
    }
}

#[inline]
/// sum of absolute values (real double-precision)
/// 
/// DASUM takes the sum of the absolute values.
pub fn dasum(n: i32, x: &[f64], incx: i32) -> f64 {
    unsafe {
        ffi::cblas_dasum(n, x.as_ptr(), incx)
    }
}

#[inline]
/// Euclidean norm (single-precision)
/// 
/// SCNRM2 returns the euclidean norm of a vector via the function
/// name, so that
/// 
/// ```text
///    SCNRM2 := sqrt( x**H*x )
/// ```
pub fn scnrm2(n: i32, x: &[c32], incx: i32) -> f32 {
    unsafe {
        ffi::cblas_scnrm2(n, x.as_ptr() as *const _, incx)
    }
}

#[inline]
/// sum of absolute values (single-precision)
/// 
/// SCASUM takes the sum of the (|Re(.)| + |Im(.)|)'s of a complex vector and
/// returns a single precision result.
pub fn scasum(n: i32, x: &[c32], incx: i32) -> f32 {
    unsafe {
        ffi::cblas_scasum(n, x.as_ptr() as *const _, incx)
    }
}

#[inline]
/// Euclidean norm (double-precision)
/// 
/// DZNRM2 returns the euclidean norm of a vector via the function
/// name, so that
/// 
/// ```text
///    DZNRM2 := sqrt( x**H*x )
/// ```
pub fn dznrm2(n: i32, x: &[c64], incx: i32) -> f64 {
    unsafe {
        ffi::cblas_dznrm2(n, x.as_ptr() as *const _, incx)
    }
}

#[inline]
/// sum of absolute values (double-precision)
/// 
/// DZASUM takes the sum of the (|Re(.)| + |Im(.)|)'s of a complex vector and
/// returns a single precision result.
pub fn dzasum(n: i32, x: &[c64], incx: i32) -> f64 {
    unsafe {
        ffi::cblas_dzasum(n, x.as_ptr() as *const _, incx)
    }
}

#[inline]
/// index of max abs value (real single-precision)
/// 
/// ISAMAX finds the index of the first element having maximum absolute value.
pub fn isamax(n: i32, x: &[f32], incx: i32) -> i32 {
    unsafe {
        ffi::cblas_isamax(n, x.as_ptr(), incx) as i32
    }
}

#[inline]
/// index of max abs value (real double-precision)
/// 
/// IDAMAX finds the index of the first element having maximum absolute value.
pub fn idamax(n: i32, x: &[f64], incx: i32) -> i32 {
    unsafe {
        ffi::cblas_idamax(n, x.as_ptr(), incx) as i32
    }
}

#[inline]
/// index of max abs value (complex single-precision)
/// 
/// ICAMAX finds the index of the first element having maximum |Re(.)| + |Im(.)|
pub fn icamax(n: i32, x: &[c32], incx: i32) -> i32 {
    unsafe {
        ffi::cblas_icamax(n, x.as_ptr() as *const _, incx) as i32
    }
}

#[inline]
/// index of max abs value (complex double-precision)
/// 
/// IZAMAX finds the index of the first element having maximum |Re(.)| + |Im(.)|
pub fn izamax(n: i32, x: &[c64], incx: i32) -> i32 {
    unsafe {
        ffi::cblas_izamax(n, x.as_ptr() as *const _, incx) as i32
    }
}

#[inline]
/// swap x and y (real single-precision)
/// 
/// interchanges two vectors.
/// uses unrolled loops for increments equal to 1.
pub fn sswap(n: i32, x: &mut [f32], incx: i32, y: &mut [f32], incy: i32) {
    unsafe {
        ffi::cblas_sswap(n, x.as_mut_ptr(), incx, y.as_mut_ptr(), incy)
    }
}

#[inline]
/// copy x into y (real single-precision)
/// 
/// SCOPY copies a vector, `x`, to a vector, `y`.
/// uses unrolled loops for increments equal to 1.
pub fn scopy(n: i32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
    unsafe {
        ffi::cblas_scopy(n, x.as_ptr(), incx, y.as_mut_ptr(), incy)
    }
}

#[inline]
/// y = alpha*x + y (real single-precision)
/// 
/// SAXPY constant times a vector plus a vector.
/// uses unrolled loops for increments equal to one.
pub fn saxpy(n: i32, alpha: f32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
    unsafe {
        ffi::cblas_saxpy(n, alpha, x.as_ptr(), incx, y.as_mut_ptr(), incy)
    }
}

#[inline]
/// swap x and y (real double-precision)
/// 
/// interchanges two vectors.
/// uses unrolled loops for increments equal one.
pub fn dswap(n: i32, x: &mut [f64], incx: i32, y: &mut [f64], incy: i32) {
    unsafe {
        ffi::cblas_dswap(n, x.as_mut_ptr(), incx, y.as_mut_ptr(), incy)
    }
}

#[inline]
/// copy x into y (real double-precision)
/// 
/// DCOPY copies a vector, `x`, to a vector, `y`.
/// uses unrolled loops for increments equal to one.
pub fn dcopy(n: i32, x: &[f64], incx: i32, y: &mut [f64], incy: i32) {
    unsafe {
        ffi::cblas_dcopy(n, x.as_ptr(), incx, y.as_mut_ptr(), incy)
    }
}

#[inline]
/// y = alpha*x + y (real double-precision)
/// 
/// DAXPY constant times a vector plus a vector.
/// uses unrolled loops for increments equal to one.
pub fn daxpy(n: i32, alpha: f64, x: &[f64], incx: i32, y: &mut [f64], incy: i32) {
    unsafe {
        ffi::cblas_daxpy(n, alpha, x.as_ptr(), incx, y.as_mut_ptr(), incy)
    }
}

#[inline]
/// swap x and y (complex single-precision)
/// 
///   CSWAP interchanges two vectors.
pub fn cswap(n: i32, x: &mut [c32], incx: i32, y: &mut [c32], incy: i32) {
    unsafe {
        ffi::cblas_cswap(n, x.as_mut_ptr() as *mut _, incx, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
/// copy x into y (complex single-precision)
/// 
/// CCOPY copies a vector `x` to a vector `y`.
pub fn ccopy(n: i32, x: &[c32], incx: i32, y: &mut [c32], incy: i32) {
    unsafe {
        ffi::cblas_ccopy(n, x.as_ptr() as *const _, incx, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
/// y = alpha*x + y (complex single-precision)
/// 
/// CAXPY constant times a vector plus a vector.
pub fn caxpy(n: i32, alpha: &[c32], x: &[c32], incx: i32, y: &mut [c32], incy: i32) {
    unsafe {
        ffi::cblas_caxpy(n, alpha.as_ptr() as *const _, x.as_ptr() as *const _, incx,
                         y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
/// swap x and y (complex double-precision)
/// 
/// ZSWAP interchanges two vectors.
pub fn zswap(n: i32, x: &mut [c64], incx: i32, y: &mut [c64], incy: i32) {
    unsafe {
        ffi::cblas_zswap(n, x.as_mut_ptr() as *mut _, incx, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
/// copy x into y (complex double-precision)
/// 
/// ZCOPY copies a vector, `x`, to a vector, `y`.
pub fn zcopy(n: i32, x: &[c64], incx: i32, y: &mut [c64], incy: i32) {
    unsafe {
        ffi::cblas_zcopy(n, x.as_ptr() as *const _, incx, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
/// y = alpha*x + y (complex double-precision)
/// 
/// ZAXPY constant times a vector plus a vector.
pub fn zaxpy(n: i32, alpha: &[c64], x: &[c64], incx: i32, y: &mut [c64], incy: i32) {
    unsafe {
        ffi::cblas_zaxpy(n, alpha.as_ptr() as *const _, x.as_ptr() as *const _, incx,
                         y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
/// setup Givens rotation (real single-precision)
/// 
/// SROTG construct givens plane rotation.
pub fn srotg(a: &mut [f32], b: &mut [f32], c: &mut f32, s: &mut [f32]) {
    unsafe {
        ffi::cblas_srotg(a.as_mut_ptr(), b.as_mut_ptr(), c, s.as_mut_ptr())
    }
}

#[inline]
/// setup modified Givens rotation (real single-precision)
/// 
/// CONSTRUCT THE MODIFIED GIVENS TRANSFORMATION MATRIX H WHICH ZEROS
/// THE SECOND COMPONENT OF THE 2-VECTOR  (SQRT(SD1)*SX1,SQRT(SD2)*>    SY2)**T.
/// WITH SPARAM(1)=SFLAG, H HAS ONE OF THE FOLLOWING FORMS..
/// 
/// SFLAG=-1.E0     SFLAG=0.E0        SFLAG=1.E0     SFLAG=-2.E0
/// 
///   (SH11  SH12)    (1.E0  SH12)    (SH11  1.E0)    (1.E0  0.E0)
/// H=(          )    (          )    (          )    (          )
///   (SH21  SH22),   (SH21  1.E0),   (-1.E0 SH22),   (0.E0  1.E0).
/// LOCATIONS 2-4 OF SPARAM CONTAIN SH11,SH21,SH12, AND SH22
/// RESPECTIVELY. (VALUES OF 1.E0, -1.E0, OR 0.E0 IMPLIED BY THE
/// VALUE OF SPARAM(1) ARE NOT STORED IN SPARAM.)
/// 
/// THE VALUES OF GAMSQ AND RGAMSQ SET IN THE DATA STATEMENT MAY BE
/// INEXACT.  THIS IS OK AS THEY ARE ONLY USED FOR TESTING THE SIZE
/// OF SD1 AND SD2.  ALL ACTUAL SCALING OF DATA IS DONE USING GAM.
pub fn srotmg(d1: &mut [f32], d2: &mut [f32], b1: &mut [f32], b2: f32, p: &mut [f32]) {
    unsafe {
        ffi::cblas_srotmg(d1.as_mut_ptr(), d2.as_mut_ptr(), b1.as_mut_ptr(), b2, p.as_mut_ptr())
    }
}

#[inline]
/// apply Givens rotation (real single-precision)
/// 
/// applies a plane rotation.
pub fn srot(n: i32, x: &mut [f32], incx: i32, y: &mut [f32], incy: i32, c: f32, s: f32) {
    unsafe {
        ffi::cblas_srot(n, x.as_mut_ptr(), incx, y.as_mut_ptr(), incy, c, s)
    }
}

#[inline]
/// apply modified Givens rotation (real single-precision)
/// 
/// APPLY THE MODIFIED GIVENS TRANSFORMATION, H, TO THE 2 BY N MATRIX
/// 
/// (SX**T) , WHERE **T INDICATES TRANSPOSE. THE ELEMENTS OF SX ARE IN
/// (SX**T)
/// 
/// SX(LX+I*INCX), I = 0 TO N-1, WHERE LX = 1 IF INCX .GE. 0, ELSE
/// LX = (-INCX)*N, AND SIMILARLY FOR SY USING USING LY AND INCY.
/// WITH SPARAM(1)=SFLAG, H HAS ONE OF THE FOLLOWING FORMS..
/// 
/// SFLAG=-1.E0     SFLAG=0.E0        SFLAG=1.E0     SFLAG=-2.E0
/// 
///   (SH11  SH12)    (1.E0  SH12)    (SH11  1.E0)    (1.E0  0.E0)
/// H=(          )    (          )    (          )    (          )
///   (SH21  SH22),   (SH21  1.E0),   (-1.E0 SH22),   (0.E0  1.E0).
/// SEE  SROTMG FOR A DESCRIPTION OF DATA STORAGE IN SPARAM.
pub fn srotm(n: i32, x: &mut [f32], incx: i32, y: &mut [f32], incy: i32, p: &[f32]) {
    unsafe {
        ffi::cblas_srotm(n, x.as_mut_ptr(), incx, y.as_mut_ptr(), incy, p.as_ptr())
    }
}

#[inline]
/// setup Givens rotation (real double-precision)
/// 
/// DROTG construct givens plane rotation.
pub fn drotg(a: &mut [f64], b: &mut [f64], c: &mut f64, s: &mut [f64]) {
    unsafe {
        ffi::cblas_drotg(a.as_mut_ptr(), b.as_mut_ptr(), c, s.as_mut_ptr())
    }
}

#[inline]
/// setup modified Givens rotation (real double-precision)
/// 
/// CONSTRUCT THE MODIFIED GIVENS TRANSFORMATION MATRIX H WHICH ZEROS
/// THE SECOND COMPONENT OF THE 2-VECTOR  (DSQRT(DD1)*DX1,DSQRT(DD2)*>    DY2)**T.
/// WITH DPARAM(1)=DFLAG, H HAS ONE OF THE FOLLOWING FORMS..
/// 
/// DFLAG=-1.D0     DFLAG=0.D0        DFLAG=1.D0     DFLAG=-2.D0
/// 
///   (DH11  DH12)    (1.D0  DH12)    (DH11  1.D0)    (1.D0  0.D0)
/// H=(          )    (          )    (          )    (          )
///   (DH21  DH22),   (DH21  1.D0),   (-1.D0 DH22),   (0.D0  1.D0).
/// LOCATIONS 2-4 OF DPARAM CONTAIN DH11, DH21, DH12, AND DH22
/// RESPECTIVELY. (VALUES OF 1.D0, -1.D0, OR 0.D0 IMPLIED BY THE
/// VALUE OF DPARAM(1) ARE NOT STORED IN DPARAM.)
/// 
/// THE VALUES OF GAMSQ AND RGAMSQ SET IN THE DATA STATEMENT MAY BE
/// INEXACT.  THIS IS OK AS THEY ARE ONLY USED FOR TESTING THE SIZE
/// OF DD1 AND DD2.  ALL ACTUAL SCALING OF DATA IS DONE USING GAM.
pub fn drotmg(d1: &mut [f64], d2: &mut [f64], b1: &mut [f64], b2: f64, p: &mut [f64]) {
    unsafe {
        ffi::cblas_drotmg(d1.as_mut_ptr(), d2.as_mut_ptr(), b1.as_mut_ptr(), b2, p.as_mut_ptr())
    }
}

#[inline]
/// apply Givens rotation (real double-precision)
/// 
/// DROT applies a plane rotation.
pub fn drot(n: i32, x: &mut [f64], incx: i32, y: &mut [f64], incy: i32, c: f64, s: f64) {
    unsafe {
        ffi::cblas_drot(n, x.as_mut_ptr(), incx, y.as_mut_ptr(), incy, c, s)
    }
}

#[inline]
/// apply modified Givens rotation (real double-precision)
/// 
/// APPLY THE MODIFIED GIVENS TRANSFORMATION, H, TO THE 2 BY N MATRIX
/// 
/// (DX**T) , WHERE **T INDICATES TRANSPOSE. THE ELEMENTS OF DX ARE IN
/// (DY**T)
/// 
/// DX(LX+I*INCX), I = 0 TO N-1, WHERE LX = 1 IF INCX .GE. 0, ELSE
/// LX = (-INCX)*N, AND SIMILARLY FOR SY USING LY AND INCY.
/// WITH DPARAM(1)=DFLAG, H HAS ONE OF THE FOLLOWING FORMS..
/// 
/// DFLAG=-1.D0     DFLAG=0.D0        DFLAG=1.D0     DFLAG=-2.D0
/// 
///   (DH11  DH12)    (1.D0  DH12)    (DH11  1.D0)    (1.D0  0.D0)
/// H=(          )    (          )    (          )    (          )
///   (DH21  DH22),   (DH21  1.D0),   (-1.D0 DH22),   (0.D0  1.D0).
/// SEE DROTMG FOR A DESCRIPTION OF DATA STORAGE IN DPARAM.
pub fn drotm(n: i32, x: &mut [f64], incx: i32, y: &mut [f64], incy: i32, p: &[f64]) {
    unsafe {
        ffi::cblas_drotm(n, x.as_mut_ptr(), incx, y.as_mut_ptr(), incy, p.as_ptr())
    }
}

#[inline]
/// x = alpha*x (real single-precision)
/// 
/// scales a vector by a constant.
/// uses unrolled loops for increment equal to 1.
pub fn sscal(n: i32, alpha: f32, x: &mut [f32], incx: i32) {
    unsafe {
        ffi::cblas_sscal(n, alpha, x.as_mut_ptr(), incx)
    }
}

#[inline]
/// x = alpha*x (real double-precision)
/// 
/// DSCAL scales a vector by a constant.
/// uses unrolled loops for increment equal to one.
pub fn dscal(n: i32, alpha: f64, x: &mut [f64], incx: i32) {
    unsafe {
        ffi::cblas_dscal(n, alpha, x.as_mut_ptr(), incx)
    }
}

#[inline]
/// x = alpha*x (complex single-precision)
/// 
/// CSCAL scales a vector by a constant.
pub fn cscal(n: i32, alpha: &[c32], x: &mut [c32], incx: i32) {
    unsafe {
        ffi::cblas_cscal(n, alpha.as_ptr() as *const _, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
/// x = alpha*x (complex double-precision)
/// 
/// ZSCAL scales a vector by a constant.
pub fn zscal(n: i32, alpha: &[c64], x: &mut [c64], incx: i32) {
    unsafe {
        ffi::cblas_zscal(n, alpha.as_ptr() as *const _, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
/// x = alpha*x, scalar alpha (complex single-precision)
/// 
/// CSSCAL scales a complex vector by a real constant.
pub fn csscal(n: i32, alpha: f32, x: &mut [c32], incx: i32) {
    unsafe {
        ffi::cblas_csscal(n, alpha, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
/// x = alpha*x, scalar alpha (complex double-precision)
/// 
/// ZDSCAL scales a vector by a constant.
pub fn zdscal(n: i32, alpha: f64, x: &mut [c64], incx: i32) {
    unsafe {
        ffi::cblas_zdscal(n, alpha, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
/// matrix-vector multiply (real single-precision)
/// 
/// SGEMV  performs one of the matrix-vector operations
/// 
/// ```text
///    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
/// ```
/// 
/// where `alpha` and `beta` are scalars, `x` and `y` are vectors and `A` is an
/// `m` by `n` matrix.
pub fn sgemv(layout: Layout, transa: Transpose, m: i32, n: i32, alpha: f32, a: &[f32], lda: i32,
             x: &[f32], incx: i32, beta: f32, y: &mut [f32], incy: i32) {

    unsafe {
        ffi::cblas_sgemv(layout.into(), transa.into(), m, n, alpha, a.as_ptr(), lda, x.as_ptr(),
                         incx, beta, y.as_mut_ptr(), incy)
    }
}

#[inline]
/// banded matrix-vector multiply (real single-precision)
/// 
/// SGBMV  performs one of the matrix-vector operations
/// 
/// ```text
///    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
/// ```
/// 
/// where `alpha` and `beta` are scalars, `x` and `y` are vectors and `A` is an
/// `m` by `n` band matrix, with `kl` sub-diagonals and `ku` super-diagonals.
pub fn sgbmv(layout: Layout, transa: Transpose, m: i32, n: i32, kl: i32, ku: i32, alpha: f32,
             a: &[f32], lda: i32, x: &[f32], incx: i32, beta: f32, y: &mut [f32], incy: i32) {

    unsafe {
        ffi::cblas_sgbmv(layout.into(), transa.into(), m, n, kl, ku, alpha, a.as_ptr(), lda,
                         x.as_ptr(), incx, beta, y.as_mut_ptr(), incy)
    }
}

#[inline]
/// triangular matrix-vector multiply (real single-precision)
/// 
/// STRMV  performs one of the matrix-vector operations
/// 
/// ```text
///    x := A*x,   or   x := A**T*x,
/// ```
/// 
/// where `x` is an `n` element vector and  `A` is an `n` by `n` unit, or non-unit,
/// upper or lower triangular matrix.
pub fn strmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, a: &[f32],
             lda: i32, x: &mut [f32], incx: i32) {

    unsafe {
        ffi::cblas_strmv(layout.into(), uplo.into(), transa.into(), diag.into(), n, a.as_ptr(),
                         lda, x.as_mut_ptr(), incx)
    }
}

#[inline]
/// triangular banded matrix-vector multiply (real single-precision)
/// 
/// STBMV  performs one of the matrix-vector operations
/// 
/// ```text
///    x := A*x,   or   x := A**T*x,
/// ```
/// 
/// where `x` is an `n` element vector and  `A` is an `n` by `n` unit, or non-unit,
/// upper or lower triangular band matrix, with ( `k` + 1 ) diagonals.
pub fn stbmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, k: i32,
             a: &[f32], lda: i32, x: &mut [f32], incx: i32) {

    unsafe {
        ffi::cblas_stbmv(layout.into(), uplo.into(), transa.into(), diag.into(), n, k, a.as_ptr(),
                         lda, x.as_mut_ptr(), incx)
    }
}

#[inline]
/// triangular packed matrix-vector multiply (real single-precision)
/// 
/// STPMV  performs one of the matrix-vector operations
/// 
/// ```text
///    x := A*x,   or   x := A**T*x,
/// ```
/// 
/// where `x` is an `n` element vector and  A is an `n` by `n` unit, or non-unit,
/// upper or lower triangular matrix, supplied in packed form.
pub fn stpmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, ap: &[f32],
             x: &mut [f32], incx: i32) {

    unsafe {
        ffi::cblas_stpmv(layout.into(), uplo.into(), transa.into(), diag.into(), n, ap.as_ptr(),
                         x.as_mut_ptr(), incx)
    }
}

#[inline]
/// solving triangular matrix problems (real single-precision)
/// 
/// STRSV  solves one of the systems of equations
/// 
/// ```text
///    A*x = b,   or   A**T*x = b,
/// ```
/// 
/// where b and `x` are `n` element vectors and `A` is an `n` by `n` unit, or
/// non-unit, upper or lower triangular matrix.
/// 
/// No test for singularity or near-singularity is included in this
/// routine. Such tests must be performed before calling this routine.
pub fn strsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, a: &[f32],
             lda: i32, x: &mut [f32], incx: i32) {

    unsafe {
        ffi::cblas_strsv(layout.into(), uplo.into(), transa.into(), diag.into(), n, a.as_ptr(),
                         lda, x.as_mut_ptr(), incx)
    }
}

#[inline]
/// solving triangular banded matrix problems (real single-precision)
/// 
/// STBSV  solves one of the systems of equations
/// 
/// ```text
///    A*x = b,   or   A**T*x = b,
/// ```
/// 
/// where b and `x` are `n` element vectors and `A` is an `n` by `n` unit, or
/// non-unit, upper or lower triangular band matrix, with ( `k` + 1 )
/// diagonals.
/// 
/// No test for singularity or near-singularity is included in this
/// routine. Such tests must be performed before calling this routine.
pub fn stbsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, k: i32,
             a: &[f32], lda: i32, x: &mut [f32], incx: i32) {

    unsafe {
        ffi::cblas_stbsv(layout.into(), uplo.into(), transa.into(), diag.into(), n, k, a.as_ptr(),
                         lda, x.as_mut_ptr(), incx)
    }
}

#[inline]
/// solving triangular packed matrix problems (real single-precision)
/// 
/// STPSV  solves one of the systems of equations
/// 
/// ```text
///    A*x = b,   or   A**T*x = b,
/// ```
/// 
/// where b and `x` are `n` element vectors and A is an `n` by `n` unit, or
/// non-unit, upper or lower triangular matrix, supplied in packed form.
/// 
/// No test for singularity or near-singularity is included in this
/// routine. Such tests must be performed before calling this routine.
pub fn stpsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, ap: &[f32],
             x: &mut [f32], incx: i32) {

    unsafe {
        ffi::cblas_stpsv(layout.into(), uplo.into(), transa.into(), diag.into(), n, ap.as_ptr(),
                         x.as_mut_ptr(), incx)
    }
}

#[inline]
/// matrix-vector multiply (real double-precision)
/// 
/// DGEMV  performs one of the matrix-vector operations
/// 
/// ```text
///    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
/// ```
/// 
/// where `alpha` and `beta` are scalars, `x` and `y` are vectors and `A` is an
/// `m` by `n` matrix.
pub fn dgemv(layout: Layout, transa: Transpose, m: i32, n: i32, alpha: f64, a: &[f64], lda: i32,
             x: &[f64], incx: i32, beta: f64, y: &mut [f64], incy: i32) {

    unsafe {
        ffi::cblas_dgemv(layout.into(), transa.into(), m, n, alpha, a.as_ptr(), lda, x.as_ptr(),
                         incx, beta, y.as_mut_ptr(), incy)
    }
}

#[inline]
/// banded matrix-vector multiply (real double-precision)
/// 
/// DGBMV  performs one of the matrix-vector operations
/// 
/// ```text
///    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
/// ```
/// 
/// where `alpha` and `beta` are scalars, `x` and `y` are vectors and `A` is an
/// `m` by `n` band matrix, with `kl` sub-diagonals and `ku` super-diagonals.
pub fn dgbmv(layout: Layout, transa: Transpose, m: i32, n: i32, kl: i32, ku: i32, alpha: f64,
             a: &[f64], lda: i32, x: &[f64], incx: i32, beta: f64, y: &mut [f64], incy: i32) {

    unsafe {
        ffi::cblas_dgbmv(layout.into(), transa.into(), m, n, kl, ku, alpha, a.as_ptr(), lda,
                         x.as_ptr(), incx, beta, y.as_mut_ptr(), incy)
    }
}

#[inline]
/// triangular matrix-vector multiply (real double-precision)
/// 
/// DTRMV  performs one of the matrix-vector operations
/// 
/// ```text
///    x := A*x,   or   x := A**T*x,
/// ```
/// 
/// where `x` is an `n` element vector and  `A` is an `n` by `n` unit, or non-unit,
/// upper or lower triangular matrix.
pub fn dtrmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, a: &[f64],
             lda: i32, x: &mut [f64], incx: i32) {

    unsafe {
        ffi::cblas_dtrmv(layout.into(), uplo.into(), transa.into(), diag.into(), n, a.as_ptr(),
                         lda, x.as_mut_ptr(), incx)
    }
}

#[inline]
/// triangular banded matrix-vector multiply (real double-precision)
/// 
/// DTBMV  performs one of the matrix-vector operations
/// 
/// ```text
///    x := A*x,   or   x := A**T*x,
/// ```
/// 
/// where `x` is an `n` element vector and  `A` is an `n` by `n` unit, or non-unit,
/// upper or lower triangular band matrix, with ( `k` + 1 ) diagonals.
pub fn dtbmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, k: i32,
             a: &[f64], lda: i32, x: &mut [f64], incx: i32) {

    unsafe {
        ffi::cblas_dtbmv(layout.into(), uplo.into(), transa.into(), diag.into(), n, k, a.as_ptr(),
                         lda, x.as_mut_ptr(), incx)
    }
}

#[inline]
/// triangular packed matrix-vector multiply (real double-precision)
/// 
/// DTPMV  performs one of the matrix-vector operations
/// 
/// ```text
///    x := A*x,   or   x := A**T*x,
/// ```
/// 
/// where `x` is an `n` element vector and  A is an `n` by `n` unit, or non-unit,
/// upper or lower triangular matrix, supplied in packed form.
pub fn dtpmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, ap: &[f64],
             x: &mut [f64], incx: i32) {

    unsafe {
        ffi::cblas_dtpmv(layout.into(), uplo.into(), transa.into(), diag.into(), n, ap.as_ptr(),
                         x.as_mut_ptr(), incx)
    }
}

#[inline]
/// solving triangular matrix problems (real double-precision)
/// 
/// DTRSV  solves one of the systems of equations
/// 
/// ```text
///    A*x = b,   or   A**T*x = b,
/// ```
/// 
/// where b and `x` are `n` element vectors and `A` is an `n` by `n` unit, or
/// non-unit, upper or lower triangular matrix.
/// 
/// No test for singularity or near-singularity is included in this
/// routine. Such tests must be performed before calling this routine.
pub fn dtrsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, a: &[f64],
             lda: i32, x: &mut [f64], incx: i32) {

    unsafe {
        ffi::cblas_dtrsv(layout.into(), uplo.into(), transa.into(), diag.into(), n, a.as_ptr(),
                         lda, x.as_mut_ptr(), incx)
    }
}

#[inline]
/// solving triangular banded matrix problems (real double-precision)
/// 
/// DTBSV  solves one of the systems of equations
/// 
/// ```text
///    A*x = b,   or   A**T*x = b,
/// ```
/// 
/// where b and `x` are `n` element vectors and `A` is an `n` by `n` unit, or
/// non-unit, upper or lower triangular band matrix, with ( `k` + 1 )
/// diagonals.
/// 
/// No test for singularity or near-singularity is included in this
/// routine. Such tests must be performed before calling this routine.
pub fn dtbsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, k: i32,
             a: &[f64], lda: i32, x: &mut [f64], incx: i32) {

    unsafe {
        ffi::cblas_dtbsv(layout.into(), uplo.into(), transa.into(), diag.into(), n, k, a.as_ptr(),
                         lda, x.as_mut_ptr(), incx)
    }
}

#[inline]
/// solving triangular packed matrix problems (real double-precision)
/// 
/// DTPSV  solves one of the systems of equations
/// 
/// ```text
///    A*x = b,   or   A**T*x = b,
/// ```
/// 
/// where b and `x` are `n` element vectors and A is an `n` by `n` unit, or
/// non-unit, upper or lower triangular matrix, supplied in packed form.
/// 
/// No test for singularity or near-singularity is included in this
/// routine. Such tests must be performed before calling this routine.
pub fn dtpsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, ap: &[f64],
             x: &mut [f64], incx: i32) {

    unsafe {
        ffi::cblas_dtpsv(layout.into(), uplo.into(), transa.into(), diag.into(), n, ap.as_ptr(),
                         x.as_mut_ptr(), incx)
    }
}

#[inline]
/// matrix-vector multiply (complex single-precision)
/// 
/// CGEMV performs one of the matrix-vector operations
/// 
/// ```text
///    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,   or
/// ```
/// 
/// ```text
///    y := alpha*A**H*x + beta*y,
/// ```
/// 
/// where `alpha` and `beta` are scalars, `x` and `y` are vectors and `A` is an
/// `m` by `n` matrix.
pub fn cgemv(layout: Layout, transa: Transpose, m: i32, n: i32, alpha: &[c32], a: &[c32], lda: i32,
             x: &[c32], incx: i32, beta: &[c32], y: &mut [c32], incy: i32) {

    unsafe {
        ffi::cblas_cgemv(layout.into(), transa.into(), m, n, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, x.as_ptr() as *const _, incx,
                         beta.as_ptr() as *const _, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
/// banded matrix-vector multiply (complex single-precision)
/// 
/// CGBMV  performs one of the matrix-vector operations
/// 
/// ```text
///    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,   or
/// ```
/// 
/// ```text
///    y := alpha*A**H*x + beta*y,
/// ```
/// 
/// where `alpha` and `beta` are scalars, `x` and `y` are vectors and `A` is an
/// `m` by `n` band matrix, with `kl` sub-diagonals and `ku` super-diagonals.
pub fn cgbmv(layout: Layout, transa: Transpose, m: i32, n: i32, kl: i32, ku: i32, alpha: &[c32],
             a: &[c32], lda: i32, x: &[c32], incx: i32, beta: &[c32], y: &mut [c32], incy: i32) {

    unsafe {
        ffi::cblas_cgbmv(layout.into(), transa.into(), m, n, kl, ku, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, x.as_ptr() as *const _, incx,
                         beta.as_ptr() as *const _, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
/// triangular matrix-vector multiply (complex single-precision)
/// 
/// CTRMV  performs one of the matrix-vector operations
/// 
/// ```text
///    x := A*x,   or   x := A**T*x,   or   x := A**H*x,
/// ```
/// 
/// where `x` is an `n` element vector and  `A` is an `n` by `n` unit, or non-unit,
/// upper or lower triangular matrix.
pub fn ctrmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, a: &[c32],
             lda: i32, x: &mut [c32], incx: i32) {

    unsafe {
        ffi::cblas_ctrmv(layout.into(), uplo.into(), transa.into(), diag.into(), n,
                         a.as_ptr() as *const _, lda, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
/// triangular banded matrix-vector multiply (complex single-precision)
/// 
/// CTBMV  performs one of the matrix-vector operations
/// 
/// ```text
///    x := A*x,   or   x := A**T*x,   or   x := A**H*x,
/// ```
/// 
/// where `x` is an `n` element vector and  `A` is an `n` by `n` unit, or non-unit,
/// upper or lower triangular band matrix, with ( `k` + 1 ) diagonals.
pub fn ctbmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, k: i32,
             a: &[c32], lda: i32, x: &mut [c32], incx: i32) {

    unsafe {
        ffi::cblas_ctbmv(layout.into(), uplo.into(), transa.into(), diag.into(), n, k,
                         a.as_ptr() as *const _, lda, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
/// triangular packed matrix-vector multiply (complex single-precision)
/// 
/// CTPMV  performs one of the matrix-vector operations
/// 
/// ```text
///    x := A*x,   or   x := A**T*x,   or   x := A**H*x,
/// ```
/// 
/// where `x` is an `n` element vector and  A is an `n` by `n` unit, or non-unit,
/// upper or lower triangular matrix, supplied in packed form.
pub fn ctpmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, ap: &[c32],
             x: &mut [c32], incx: i32) {

    unsafe {
        ffi::cblas_ctpmv(layout.into(), uplo.into(), transa.into(), diag.into(), n,
                         ap.as_ptr() as *const _, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
/// solving triangular matrix problems (complex single-precision)
/// 
/// CTRSV  solves one of the systems of equations
/// 
/// ```text
///    A*x = b,   or   A**T*x = b,   or   A**H*x = b,
/// ```
/// 
/// where b and `x` are `n` element vectors and `A` is an `n` by `n` unit, or
/// non-unit, upper or lower triangular matrix.
/// 
/// No test for singularity or near-singularity is included in this
/// routine. Such tests must be performed before calling this routine.
pub fn ctrsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, a: &[c32],
             lda: i32, x: &mut [c32], incx: i32) {

    unsafe {
        ffi::cblas_ctrsv(layout.into(), uplo.into(), transa.into(), diag.into(), n,
                         a.as_ptr() as *const _, lda, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
/// solving triangular banded matrix problems (complex single-precision)
/// 
/// CTBSV  solves one of the systems of equations
/// 
/// ```text
///    A*x = b,   or   A**T*x = b,   or   A**H*x = b,
/// ```
/// 
/// where b and `x` are `n` element vectors and `A` is an `n` by `n` unit, or
/// non-unit, upper or lower triangular band matrix, with ( `k` + 1 )
/// diagonals.
/// 
/// No test for singularity or near-singularity is included in this
/// routine. Such tests must be performed before calling this routine.
pub fn ctbsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, k: i32,
             a: &[c32], lda: i32, x: &mut [c32], incx: i32) {

    unsafe {
        ffi::cblas_ctbsv(layout.into(), uplo.into(), transa.into(), diag.into(), n, k,
                         a.as_ptr() as *const _, lda, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
/// solving triangular packed matrix problems (complex single-precision)
/// 
/// CTPSV  solves one of the systems of equations
/// 
/// ```text
///    A*x = b,   or   A**T*x = b,   or   A**H*x = b,
/// ```
/// 
/// where b and `x` are `n` element vectors and A is an `n` by `n` unit, or
/// non-unit, upper or lower triangular matrix, supplied in packed form.
/// 
/// No test for singularity or near-singularity is included in this
/// routine. Such tests must be performed before calling this routine.
pub fn ctpsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, ap: &[c32],
             x: &mut [c32], incx: i32) {

    unsafe {
        ffi::cblas_ctpsv(layout.into(), uplo.into(), transa.into(), diag.into(), n,
                         ap.as_ptr() as *const _, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
/// matrix-vector multiply (complex double-precision)
/// 
/// ZGEMV  performs one of the matrix-vector operations
/// 
/// ```text
///    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,   or
/// ```
/// 
/// ```text
///    y := alpha*A**H*x + beta*y,
/// ```
/// 
/// where `alpha` and `beta` are scalars, `x` and `y` are vectors and `A` is an
/// `m` by `n` matrix.
pub fn zgemv(layout: Layout, transa: Transpose, m: i32, n: i32, alpha: &[c64], a: &[c64], lda: i32,
             x: &[c64], incx: i32, beta: &[c64], y: &mut [c64], incy: i32) {

    unsafe {
        ffi::cblas_zgemv(layout.into(), transa.into(), m, n, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, x.as_ptr() as *const _, incx,
                         beta.as_ptr() as *const _, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
/// banded matrix-vector multiply (complex double-precision)
/// 
/// ZGBMV  performs one of the matrix-vector operations
/// 
/// ```text
///    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,   or
/// ```
/// 
/// ```text
///    y := alpha*A**H*x + beta*y,
/// ```
/// 
/// where `alpha` and `beta` are scalars, `x` and `y` are vectors and `A` is an
/// `m` by `n` band matrix, with `kl` sub-diagonals and `ku` super-diagonals.
pub fn zgbmv(layout: Layout, transa: Transpose, m: i32, n: i32, kl: i32, ku: i32, alpha: &[c64],
             a: &[c64], lda: i32, x: &[c64], incx: i32, beta: &[c64], y: &mut [c64], incy: i32) {

    unsafe {
        ffi::cblas_zgbmv(layout.into(), transa.into(), m, n, kl, ku, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, x.as_ptr() as *const _, incx,
                         beta.as_ptr() as *const _, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
/// triangular matrix-vector multiply (complex double-precision)
/// 
/// ZTRMV  performs one of the matrix-vector operations
/// 
/// ```text
///    x := A*x,   or   x := A**T*x,   or   x := A**H*x,
/// ```
/// 
/// where `x` is an `n` element vector and  `A` is an `n` by `n` unit, or non-unit,
/// upper or lower triangular matrix.
pub fn ztrmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, a: &[c64],
             lda: i32, x: &mut [c64], incx: i32) {

    unsafe {
        ffi::cblas_ztrmv(layout.into(), uplo.into(), transa.into(), diag.into(), n,
                         a.as_ptr() as *const _, lda, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
/// triangular banded matrix-vector multiply (complex double-precision)
/// 
/// ZTBMV  performs one of the matrix-vector operations
/// 
/// ```text
///    x := A*x,   or   x := A**T*x,   or   x := A**H*x,
/// ```
/// 
/// where `x` is an `n` element vector and  `A` is an `n` by `n` unit, or non-unit,
/// upper or lower triangular band matrix, with ( `k` + 1 ) diagonals.
pub fn ztbmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, k: i32,
             a: &[c64], lda: i32, x: &mut [c64], incx: i32) {

    unsafe {
        ffi::cblas_ztbmv(layout.into(), uplo.into(), transa.into(), diag.into(), n, k,
                         a.as_ptr() as *const _, lda, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
/// triangular packed matrix-vector multiply (complex double-precision)
/// 
/// ZTPMV  performs one of the matrix-vector operations
/// 
/// ```text
///    x := A*x,   or   x := A**T*x,   or   x := A**H*x,
/// ```
/// 
/// where `x` is an `n` element vector and  A is an `n` by `n` unit, or non-unit,
/// upper or lower triangular matrix, supplied in packed form.
pub fn ztpmv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, ap: &[c64],
             x: &mut [c64], incx: i32) {

    unsafe {
        ffi::cblas_ztpmv(layout.into(), uplo.into(), transa.into(), diag.into(), n,
                         ap.as_ptr() as *const _, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
/// solving triangular matrix problems (complex double-precision)
/// 
/// ZTRSV  solves one of the systems of equations
/// 
/// ```text
///    A*x = b,   or   A**T*x = b,   or   A**H*x = b,
/// ```
/// 
/// where b and `x` are `n` element vectors and `A` is an `n` by `n` unit, or
/// non-unit, upper or lower triangular matrix.
/// 
/// No test for singularity or near-singularity is included in this
/// routine. Such tests must be performed before calling this routine.
pub fn ztrsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, a: &[c64],
             lda: i32, x: &mut [c64], incx: i32) {

    unsafe {
        ffi::cblas_ztrsv(layout.into(), uplo.into(), transa.into(), diag.into(), n,
                         a.as_ptr() as *const _, lda, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
/// solving triangular banded matrix problems (complex double-precision)
/// 
/// ZTBSV  solves one of the systems of equations
/// 
/// ```text
///    A*x = b,   or   A**T*x = b,   or   A**H*x = b,
/// ```
/// 
/// where b and `x` are `n` element vectors and `A` is an `n` by `n` unit, or
/// non-unit, upper or lower triangular band matrix, with ( `k` + 1 )
/// diagonals.
/// 
/// No test for singularity or near-singularity is included in this
/// routine. Such tests must be performed before calling this routine.
pub fn ztbsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, k: i32,
             a: &[c64], lda: i32, x: &mut [c64], incx: i32) {

    unsafe {
        ffi::cblas_ztbsv(layout.into(), uplo.into(), transa.into(), diag.into(), n, k,
                         a.as_ptr() as *const _, lda, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
/// solving triangular packed matrix problems (complex double-precision)
/// 
/// ZTPSV  solves one of the systems of equations
/// 
/// ```text
///    A*x = b,   or   A**T*x = b,   or   A**H*x = b,
/// ```
/// 
/// where b and `x` are `n` element vectors and A is an `n` by `n` unit, or
/// non-unit, upper or lower triangular matrix, supplied in packed form.
/// 
/// No test for singularity or near-singularity is included in this
/// routine. Such tests must be performed before calling this routine.
pub fn ztpsv(layout: Layout, uplo: Part, transa: Transpose, diag: Diagonal, n: i32, ap: &[c64],
             x: &mut [c64], incx: i32) {

    unsafe {
        ffi::cblas_ztpsv(layout.into(), uplo.into(), transa.into(), diag.into(), n,
                         ap.as_ptr() as *const _, x.as_mut_ptr() as *mut _, incx)
    }
}

#[inline]
/// symmetric matrix-vector multiply (real single-precision)
/// 
/// SSYMV  performs the matrix-vector  operation
/// 
/// ```text
///    y := alpha*A*x + beta*y,
/// ```
/// 
/// where `alpha` and `beta` are scalars, `x` and `y` are `n` element vectors and
/// `A` is an `n` by `n` symmetric matrix.
pub fn ssymv(layout: Layout, uplo: Part, n: i32, alpha: f32, a: &[f32], lda: i32, x: &[f32],
             incx: i32, beta: f32, y: &mut [f32], incy: i32) {

    unsafe {
        ffi::cblas_ssymv(layout.into(), uplo.into(), n, alpha, a.as_ptr(), lda, x.as_ptr(), incx,
                         beta, y.as_mut_ptr(), incy)
    }
}

#[inline]
/// symmetric banded matrix-vector multiply (real single-precision)
/// 
/// SSBMV  performs the matrix-vector  operation
/// 
/// ```text
///    y := alpha*A*x + beta*y,
/// ```
/// 
/// where `alpha` and `beta` are scalars, `x` and `y` are `n` element vectors and
/// `A` is an `n` by `n` symmetric band matrix, with `k` super-diagonals.
pub fn ssbmv(layout: Layout, uplo: Part, n: i32, k: i32, alpha: f32, a: &[f32], lda: i32,
             x: &[f32], incx: i32, beta: f32, y: &mut [f32], incy: i32) {

    unsafe {
        ffi::cblas_ssbmv(layout.into(), uplo.into(), n, k, alpha, a.as_ptr(), lda, x.as_ptr(),
                         incx, beta, y.as_mut_ptr(), incy)
    }
}

#[inline]
/// symmetric packed matrix-vector multiply (real single-precision)
/// 
/// SSPMV  performs the matrix-vector operation
/// 
/// ```text
///    y := alpha*A*x + beta*y,
/// ```
/// 
/// where `alpha` and `beta` are scalars, `x` and `y` are `n` element vectors and
/// A is an `n` by `n` symmetric matrix, supplied in packed form.
pub fn sspmv(layout: Layout, uplo: Part, n: i32, alpha: f32, ap: &[f32], x: &[f32], incx: i32,
             beta: f32, y: &mut [f32], incy: i32) {

    unsafe {
        ffi::cblas_sspmv(layout.into(), uplo.into(), n, alpha, ap.as_ptr(), x.as_ptr(), incx, beta,
                         y.as_mut_ptr(), incy)
    }
}

#[inline]
/// performs the rank 1 operation A := alpha*x*y' + A (real single-precision)
/// 
/// SGER   performs the rank 1 operation
/// 
/// ```text
///    A := alpha*x*y**T + A,
/// ```
/// 
/// where `alpha` is a scalar, `x` is an `m` element vector, `y` is an `n` element
/// vector and `A` is an `m` by `n` matrix.
pub fn sger(layout: Layout, m: i32, n: i32, alpha: f32, x: &[f32], incx: i32, y: &[f32], incy: i32,
            a: &mut [f32], lda: i32) {

    unsafe {
        ffi::cblas_sger(layout.into(), m, n, alpha, x.as_ptr(), incx, y.as_ptr(), incy,
                        a.as_mut_ptr(), lda)
    }
}

#[inline]
/// performs the symmetric rank 1 operation A := alpha*x*x' + A (real single-precision)
/// 
/// SSYR   performs the symmetric rank 1 operation
/// 
/// ```text
///    A := alpha*x*x**T + A,
/// ```
/// 
/// where `alpha` is a real scalar, `x` is an `n` element vector and `A` is an
/// `n` by `n` symmetric matrix.
pub fn ssyr(layout: Layout, uplo: Part, n: i32, alpha: f32, x: &[f32], incx: i32, a: &mut [f32],
            lda: i32) {

    unsafe {
        ffi::cblas_ssyr(layout.into(), uplo.into(), n, alpha, x.as_ptr(), incx, a.as_mut_ptr(),
                        lda)
    }
}

#[inline]
/// symmetric packed rank 1 operation A := alpha*x*x' + A (real single-precision)
/// 
/// SSPR    performs the symmetric rank 1 operation
/// 
/// ```text
///    A := alpha*x*x**T + A,
/// ```
/// 
/// where `alpha` is a real scalar, `x` is an `n` element vector and A is an
/// `n` by `n` symmetric matrix, supplied in packed form.
pub fn sspr(layout: Layout, uplo: Part, n: i32, alpha: f32, x: &[f32], incx: i32, ap: &mut [f32]) {
    unsafe {
        ffi::cblas_sspr(layout.into(), uplo.into(), n, alpha, x.as_ptr(), incx, ap.as_mut_ptr())
    }
}

#[inline]
/// performs the symmetric rank 2 operation, A := alpha*x*y' + alpha*y*x' + A (real single-precision)
/// 
/// SSYR2  performs the symmetric rank 2 operation
/// 
/// ```text
///    A := alpha*x*y**T + alpha*y*x**T + A,
/// ```
/// 
/// where `alpha` is a scalar, `x` and `y` are `n` element vectors and `A` is an `n`
/// by `n` symmetric matrix.
pub fn ssyr2(layout: Layout, uplo: Part, n: i32, alpha: f32, x: &[f32], incx: i32, y: &[f32],
             incy: i32, a: &mut [f32], lda: i32) {

    unsafe {
        ffi::cblas_ssyr2(layout.into(), uplo.into(), n, alpha, x.as_ptr(), incx, y.as_ptr(), incy,
                         a.as_mut_ptr(), lda)
    }
}

#[inline]
/// performs the symmetric packed rank 2 operation, A := alpha*x*y' + alpha*y*x' + A (real single-precision)
/// 
/// SSPR2  performs the symmetric rank 2 operation
/// 
/// ```text
///    A := alpha*x*y**T + alpha*y*x**T + A,
/// ```
/// 
/// where `alpha` is a scalar, `x` and `y` are `n` element vectors and `A` is an
/// `n` by `n` symmetric matrix, supplied in packed form.
pub fn sspr2(layout: Layout, uplo: Part, n: i32, alpha: f32, x: &[f32], incx: i32, y: &[f32],
             incy: i32, a: &mut [f32]) {

    unsafe {
        ffi::cblas_sspr2(layout.into(), uplo.into(), n, alpha, x.as_ptr(), incx, y.as_ptr(), incy,
                         a.as_mut_ptr())
    }
}

#[inline]
/// symmetric matrix-vector multiply (real double-precision)
/// 
/// DSYMV  performs the matrix-vector  operation
/// 
/// ```text
///    y := alpha*A*x + beta*y,
/// ```
/// 
/// where `alpha` and `beta` are scalars, `x` and `y` are `n` element vectors and
/// `A` is an `n` by `n` symmetric matrix.
pub fn dsymv(layout: Layout, uplo: Part, n: i32, alpha: f64, a: &[f64], lda: i32, x: &[f64],
             incx: i32, beta: f64, y: &mut [f64], incy: i32) {

    unsafe {
        ffi::cblas_dsymv(layout.into(), uplo.into(), n, alpha, a.as_ptr(), lda, x.as_ptr(), incx,
                         beta, y.as_mut_ptr(), incy)
    }
}

#[inline]
/// symmetric banded matrix-vector multiply (real double-precision)
/// 
/// DSBMV  performs the matrix-vector  operation
/// 
/// ```text
///    y := alpha*A*x + beta*y,
/// ```
/// 
/// where `alpha` and `beta` are scalars, `x` and `y` are `n` element vectors and
/// `A` is an `n` by `n` symmetric band matrix, with `k` super-diagonals.
pub fn dsbmv(layout: Layout, uplo: Part, n: i32, k: i32, alpha: f64, a: &[f64], lda: i32,
             x: &[f64], incx: i32, beta: f64, y: &mut [f64], incy: i32) {

    unsafe {
        ffi::cblas_dsbmv(layout.into(), uplo.into(), n, k, alpha, a.as_ptr(), lda, x.as_ptr(),
                         incx, beta, y.as_mut_ptr(), incy)
    }
}

#[inline]
/// symmetric packed matrix-vector multiply (real double-precision)
/// 
/// DSPMV  performs the matrix-vector operation
/// 
/// ```text
///    y := alpha*A*x + beta*y,
/// ```
/// 
/// where `alpha` and `beta` are scalars, `x` and `y` are `n` element vectors and
/// A is an `n` by `n` symmetric matrix, supplied in packed form.
pub fn dspmv(layout: Layout, uplo: Part, n: i32, alpha: f64, ap: &[f64], x: &[f64], incx: i32,
             beta: f64, y: &mut [f64], incy: i32) {

    unsafe {
        ffi::cblas_dspmv(layout.into(), uplo.into(), n, alpha, ap.as_ptr(), x.as_ptr(), incx, beta,
                         y.as_mut_ptr(), incy)
    }
}

#[inline]
/// performs the rank 1 operation A := alpha*x*y' + A (real double-precision)
/// 
/// DGER   performs the rank 1 operation
/// 
/// ```text
///    A := alpha*x*y**T + A,
/// ```
/// 
/// where `alpha` is a scalar, `x` is an `m` element vector, `y` is an `n` element
/// vector and `A` is an `m` by `n` matrix.
pub fn dger(layout: Layout, m: i32, n: i32, alpha: f64, x: &[f64], incx: i32, y: &[f64], incy: i32,
            a: &mut [f64], lda: i32) {

    unsafe {
        ffi::cblas_dger(layout.into(), m, n, alpha, x.as_ptr(), incx, y.as_ptr(), incy,
                        a.as_mut_ptr(), lda)
    }
}

#[inline]
/// performs the symmetric rank 1 operation A := alpha*x*x' + A (real double-precision)
/// 
/// DSYR   performs the symmetric rank 1 operation
/// 
/// ```text
///    A := alpha*x*x**T + A,
/// ```
/// 
/// where `alpha` is a real scalar, `x` is an `n` element vector and `A` is an
/// `n` by `n` symmetric matrix.
pub fn dsyr(layout: Layout, uplo: Part, n: i32, alpha: f64, x: &[f64], incx: i32, a: &mut [f64],
            lda: i32) {

    unsafe {
        ffi::cblas_dsyr(layout.into(), uplo.into(), n, alpha, x.as_ptr(), incx, a.as_mut_ptr(),
                        lda)
    }
}

#[inline]
/// symmetric packed rank 1 operation A := alpha*x*x' + A (real double-precision)
/// 
/// DSPR    performs the symmetric rank 1 operation
/// 
/// ```text
///    A := alpha*x*x**T + A,
/// ```
/// 
/// where `alpha` is a real scalar, `x` is an `n` element vector and A is an
/// `n` by `n` symmetric matrix, supplied in packed form.
pub fn dspr(layout: Layout, uplo: Part, n: i32, alpha: f64, x: &[f64], incx: i32, ap: &mut [f64]) {
    unsafe {
        ffi::cblas_dspr(layout.into(), uplo.into(), n, alpha, x.as_ptr(), incx, ap.as_mut_ptr())
    }
}

#[inline]
/// performs the symmetric rank 2 operation, A := alpha*x*y' + alpha*y*x' + A (real double-precision)
/// 
/// DSYR2  performs the symmetric rank 2 operation
/// 
/// ```text
///    A := alpha*x*y**T + alpha*y*x**T + A,
/// ```
/// 
/// where `alpha` is a scalar, `x` and `y` are `n` element vectors and `A` is an `n`
/// by `n` symmetric matrix.
pub fn dsyr2(layout: Layout, uplo: Part, n: i32, alpha: f64, x: &[f64], incx: i32, y: &[f64],
             incy: i32, a: &mut [f64], lda: i32) {

    unsafe {
        ffi::cblas_dsyr2(layout.into(), uplo.into(), n, alpha, x.as_ptr(), incx, y.as_ptr(), incy,
                         a.as_mut_ptr(), lda)
    }
}

#[inline]
/// performs the symmetric packed rank 2 operation, A := alpha*x*y' + alpha*y*x' + A (real double-precision)
/// 
/// DSPR2  performs the symmetric rank 2 operation
/// 
/// ```text
///    A := alpha*x*y**T + alpha*y*x**T + A,
/// ```
/// 
/// where `alpha` is a scalar, `x` and `y` are `n` element vectors and `A` is an
/// `n` by `n` symmetric matrix, supplied in packed form.
pub fn dspr2(layout: Layout, uplo: Part, n: i32, alpha: f64, x: &[f64], incx: i32, y: &[f64],
             incy: i32, a: &mut [f64]) {

    unsafe {
        ffi::cblas_dspr2(layout.into(), uplo.into(), n, alpha, x.as_ptr(), incx, y.as_ptr(), incy,
                         a.as_mut_ptr())
    }
}

#[inline]
/// hermitian matrix-vector multiply (complex single-precision)
/// 
/// CHEMV  performs the matrix-vector  operation
/// 
/// ```text
///    y := alpha*A*x + beta*y,
/// ```
/// 
/// where `alpha` and `beta` are scalars, `x` and `y` are `n` element vectors and
/// `A` is an `n` by `n` hermitian matrix.
pub fn chemv(layout: Layout, uplo: Part, n: i32, alpha: &[c32], a: &[c32], lda: i32, x: &[c32],
             incx: i32, beta: &[c32], y: &mut [c32], incy: i32) {

    unsafe {
        ffi::cblas_chemv(layout.into(), uplo.into(), n, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, x.as_ptr() as *const _, incx,
                         beta.as_ptr() as *const _, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
/// hermitian banded matrix-vector multiply (complex single-precision)
/// 
/// CHBMV  performs the matrix-vector  operation
/// 
/// ```text
///    y := alpha*A*x + beta*y,
/// ```
/// 
/// where `alpha` and `beta` are scalars, `x` and `y` are `n` element vectors and
/// `A` is an `n` by `n` hermitian band matrix, with `k` super-diagonals.
pub fn chbmv(layout: Layout, uplo: Part, n: i32, k: i32, alpha: &[c32], a: &[c32], lda: i32,
             x: &[c32], incx: i32, beta: &[c32], y: &mut [c32], incy: i32) {

    unsafe {
        ffi::cblas_chbmv(layout.into(), uplo.into(), n, k, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, x.as_ptr() as *const _, incx,
                         beta.as_ptr() as *const _, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
/// hermitian packed matrix-vector multiply (complex single-precision)
/// 
/// CHPMV  performs the matrix-vector operation
/// 
/// ```text
///    y := alpha*A*x + beta*y,
/// ```
/// 
/// where `alpha` and `beta` are scalars, `x` and `y` are `n` element vectors and
/// A is an `n` by `n` hermitian matrix, supplied in packed form.
pub fn chpmv(layout: Layout, uplo: Part, n: i32, alpha: &[c32], ap: &[c32], x: &[c32], incx: i32,
             beta: &[c32], y: &mut [c32], incy: i32) {

    unsafe {
        ffi::cblas_chpmv(layout.into(), uplo.into(), n, alpha.as_ptr() as *const _,
                         ap.as_ptr() as *const _, x.as_ptr() as *const _, incx,
                         beta.as_ptr() as *const _, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
/// performs the rank 1 operation A := alpha*x*y' + A (complex single-precision)
/// 
/// CGERU  performs the rank 1 operation
/// 
/// ```text
///    A := alpha*x*y**T + A,
/// ```
/// 
/// where `alpha` is a scalar, `x` is an `m` element vector, `y` is an `n` element
/// vector and `A` is an `m` by `n` matrix.
pub fn cgeru(layout: Layout, m: i32, n: i32, alpha: &[c32], x: &[c32], incx: i32, y: &[c32],
             incy: i32, a: &mut [c32], lda: i32) {

    unsafe {
        ffi::cblas_cgeru(layout.into(), m, n, alpha.as_ptr() as *const _, x.as_ptr() as *const _,
                         incx, y.as_ptr() as *const _, incy, a.as_mut_ptr() as *mut _, lda)
    }
}

#[inline]
/// performs the rank 1 operation A := alpha*x*conjg( y' ) + A (complex single-precision)
/// 
/// CGERC  performs the rank 1 operation
/// 
/// ```text
///    A := alpha*x*y**H + A,
/// ```
/// 
/// where `alpha` is a scalar, `x` is an `m` element vector, `y` is an `n` element
/// vector and `A` is an `m` by `n` matrix.
pub fn cgerc(layout: Layout, m: i32, n: i32, alpha: &[c32], x: &[c32], incx: i32, y: &[c32],
             incy: i32, a: &mut [c32], lda: i32) {

    unsafe {
        ffi::cblas_cgerc(layout.into(), m, n, alpha.as_ptr() as *const _, x.as_ptr() as *const _,
                         incx, y.as_ptr() as *const _, incy, a.as_mut_ptr() as *mut _, lda)
    }
}

#[inline]
/// hermitian rank 1 operation A := alpha*x*conjg(x') + A (complex single-precision)
/// 
/// CHER   performs the hermitian rank 1 operation
/// 
/// ```text
///    A := alpha*x*x**H + A,
/// ```
/// 
/// where `alpha` is a real scalar, `x` is an `n` element vector and `A` is an
/// `n` by `n` hermitian matrix.
pub fn cher(layout: Layout, uplo: Part, n: i32, alpha: f32, x: &[c32], incx: i32, a: &mut [c32],
            lda: i32) {

    unsafe {
        ffi::cblas_cher(layout.into(), uplo.into(), n, alpha, x.as_ptr() as *const _, incx,
                        a.as_mut_ptr() as *mut _, lda)
    }
}

#[inline]
/// hermitian packed rank 1 operation A := alpha*x*conjg( x' ) + A (complex single-precision)
/// 
/// CHPR    performs the hermitian rank 1 operation
/// 
/// ```text
///    A := alpha*x*x**H + A,
/// ```
/// 
/// where `alpha` is a real scalar, `x` is an `n` element vector and `A` is an
/// `n` by `n` hermitian matrix, supplied in packed form.
pub fn chpr(layout: Layout, uplo: Part, n: i32, alpha: f32, x: &[c32], incx: i32, a: &mut [c32]) {
    unsafe {
        ffi::cblas_chpr(layout.into(), uplo.into(), n, alpha, x.as_ptr() as *const _, incx,
                        a.as_mut_ptr() as *mut _)
    }
}

#[inline]
/// hermitian rank 2 operation (complex single-precision)
/// 
/// CHER2  performs the hermitian rank 2 operation
/// 
/// ```text
///    A := alpha*x*y**H + conjg( alpha )*y*x**H + A,
/// ```
/// 
/// where `alpha` is a scalar, `x` and `y` are `n` element vectors and `A` is an `n`
/// by `n` hermitian matrix.
pub fn cher2(layout: Layout, uplo: Part, n: i32, alpha: &[c32], x: &[c32], incx: i32, y: &[c32],
             incy: i32, a: &mut [c32], lda: i32) {

    unsafe {
        ffi::cblas_cher2(layout.into(), uplo.into(), n, alpha.as_ptr() as *const _,
                         x.as_ptr() as *const _, incx, y.as_ptr() as *const _, incy,
                         a.as_mut_ptr() as *mut _, lda)
    }
}

#[inline]
/// hermitian packed rank 2 operation (complex single-precision)
/// 
/// CHPR2  performs the hermitian rank 2 operation
/// 
/// ```text
///    A := alpha*x*y**H + conjg( alpha )*y*x**H + A,
/// ```
/// 
/// where `alpha` is a scalar, `x` and `y` are `n` element vectors and A is an
/// `n` by `n` hermitian matrix, supplied in packed form.
pub fn chpr2(layout: Layout, uplo: Part, n: i32, alpha: &[c32], x: &[c32], incx: i32, y: &[c32],
             incy: i32, ap: &mut [c32]) {

    unsafe {
        ffi::cblas_chpr2(layout.into(), uplo.into(), n, alpha.as_ptr() as *const _,
                         x.as_ptr() as *const _, incx, y.as_ptr() as *const _, incy,
                         ap.as_mut_ptr() as *mut _)
    }
}

#[inline]
/// hermitian matrix-vector multiply (complex double-precision)
/// 
/// ZHEMV  performs the matrix-vector  operation
/// 
/// ```text
///    y := alpha*A*x + beta*y,
/// ```
/// 
/// where `alpha` and `beta` are scalars, `x` and `y` are `n` element vectors and
/// `A` is an `n` by `n` hermitian matrix.
pub fn zhemv(layout: Layout, uplo: Part, n: i32, alpha: &[c64], a: &[c64], lda: i32, x: &[c64],
             incx: i32, beta: &[c64], y: &mut [c64], incy: i32) {

    unsafe {
        ffi::cblas_zhemv(layout.into(), uplo.into(), n, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, x.as_ptr() as *const _, incx,
                         beta.as_ptr() as *const _, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
/// hermitian banded matrix-vector multiply (complex double-precision)
/// 
/// ZHBMV  performs the matrix-vector  operation
/// 
/// ```text
///    y := alpha*A*x + beta*y,
/// ```
/// 
/// where `alpha` and `beta` are scalars, `x` and `y` are `n` element vectors and
/// `A` is an `n` by `n` hermitian band matrix, with `k` super-diagonals.
pub fn zhbmv(layout: Layout, uplo: Part, n: i32, k: i32, alpha: &[c64], a: &[c64], lda: i32,
             x: &[c64], incx: i32, beta: &[c64], y: &mut [c64], incy: i32) {

    unsafe {
        ffi::cblas_zhbmv(layout.into(), uplo.into(), n, k, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, x.as_ptr() as *const _, incx,
                         beta.as_ptr() as *const _, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
/// hermitian packed matrix-vector multiply (complex double-precision)
/// 
/// ZHPMV  performs the matrix-vector operation
/// 
/// ```text
///    y := alpha*A*x + beta*y,
/// ```
/// 
/// where `alpha` and `beta` are scalars, `x` and `y` are `n` element vectors and
/// A is an `n` by `n` hermitian matrix, supplied in packed form.
pub fn zhpmv(layout: Layout, uplo: Part, n: i32, alpha: &[c64], ap: &[c64], x: &[c64], incx: i32,
             beta: &[c64], y: &mut [c64], incy: i32) {

    unsafe {
        ffi::cblas_zhpmv(layout.into(), uplo.into(), n, alpha.as_ptr() as *const _,
                         ap.as_ptr() as *const _, x.as_ptr() as *const _, incx,
                         beta.as_ptr() as *const _, y.as_mut_ptr() as *mut _, incy)
    }
}

#[inline]
/// performs the rank 1 operation A := alpha*x*y' + A (complex double-precision)
/// 
/// ZGERU  performs the rank 1 operation
/// 
/// ```text
///    A := alpha*x*y**T + A,
/// ```
/// 
/// where `alpha` is a scalar, `x` is an `m` element vector, `y` is an `n` element
/// vector and `A` is an `m` by `n` matrix.
pub fn zgeru(layout: Layout, m: i32, n: i32, alpha: &[c64], x: &[c64], incx: i32, y: &[c64],
             incy: i32, a: &mut [c64], lda: i32) {

    unsafe {
        ffi::cblas_zgeru(layout.into(), m, n, alpha.as_ptr() as *const _, x.as_ptr() as *const _,
                         incx, y.as_ptr() as *const _, incy, a.as_mut_ptr() as *mut _, lda)
    }
}

#[inline]
/// performs the rank 1 operation A := alpha*x*conjg( y' ) + A (complex double-precision)
/// 
/// ZGERC  performs the rank 1 operation
/// 
/// ```text
///    A := alpha*x*y**H + A,
/// ```
/// 
/// where `alpha` is a scalar, `x` is an `m` element vector, `y` is an `n` element
/// vector and `A` is an `m` by `n` matrix.
pub fn zgerc(layout: Layout, m: i32, n: i32, alpha: &[c64], x: &[c64], incx: i32, y: &[c64],
             incy: i32, a: &mut [c64], lda: i32) {

    unsafe {
        ffi::cblas_zgerc(layout.into(), m, n, alpha.as_ptr() as *const _, x.as_ptr() as *const _,
                         incx, y.as_ptr() as *const _, incy, a.as_mut_ptr() as *mut _, lda)
    }
}

#[inline]
/// hermitian rank 1 operation A := alpha*x*conjg(x') + A (complex double-precision)
/// 
/// ZHER   performs the hermitian rank 1 operation
/// 
/// ```text
///    A := alpha*x*x**H + A,
/// ```
/// 
/// where `alpha` is a real scalar, `x` is an `n` element vector and `A` is an
/// `n` by `n` hermitian matrix.
pub fn zher(layout: Layout, uplo: Part, n: i32, alpha: f64, x: &[c64], incx: i32, a: &mut [c64],
            lda: i32) {

    unsafe {
        ffi::cblas_zher(layout.into(), uplo.into(), n, alpha, x.as_ptr() as *const _, incx,
                        a.as_mut_ptr() as *mut _, lda)
    }
}

#[inline]
/// hermitian packed rank 1 operation A := alpha*x*conjg( x' ) + A (complex double-precision)
/// 
/// ZHPR    performs the hermitian rank 1 operation
/// 
/// ```text
///    A := alpha*x*x**H + A,
/// ```
/// 
/// where `alpha` is a real scalar, `x` is an `n` element vector and `A` is an
/// `n` by `n` hermitian matrix, supplied in packed form.
pub fn zhpr(layout: Layout, uplo: Part, n: i32, alpha: f64, x: &[c64], incx: i32, a: &mut [c64]) {
    unsafe {
        ffi::cblas_zhpr(layout.into(), uplo.into(), n, alpha, x.as_ptr() as *const _, incx,
                        a.as_mut_ptr() as *mut _)
    }
}

#[inline]
/// hermitian rank 2 operation (complex double-precision)
/// 
/// ZHER2  performs the hermitian rank 2 operation
/// 
/// ```text
///    A := alpha*x*y**H + conjg( alpha )*y*x**H + A,
/// ```
/// 
/// where `alpha` is a scalar, `x` and `y` are `n` element vectors and `A` is an `n`
/// by `n` hermitian matrix.
pub fn zher2(layout: Layout, uplo: Part, n: i32, alpha: &[c64], x: &[c64], incx: i32, y: &[c64],
             incy: i32, a: &mut [c64], lda: i32) {

    unsafe {
        ffi::cblas_zher2(layout.into(), uplo.into(), n, alpha.as_ptr() as *const _,
                         x.as_ptr() as *const _, incx, y.as_ptr() as *const _, incy,
                         a.as_mut_ptr() as *mut _, lda)
    }
}

#[inline]
/// hermitian packed rank 2 operation (complex double-precision)
/// 
/// ZHPR2  performs the hermitian rank 2 operation
/// 
/// ```text
///    A := alpha*x*y**H + conjg( alpha )*y*x**H + A,
/// ```
/// 
/// where `alpha` is a scalar, `x` and `y` are `n` element vectors and A is an
/// `n` by `n` hermitian matrix, supplied in packed form.
pub fn zhpr2(layout: Layout, uplo: Part, n: i32, alpha: &[c64], x: &[c64], incx: i32, y: &[c64],
             incy: i32, ap: &mut [c64]) {

    unsafe {
        ffi::cblas_zhpr2(layout.into(), uplo.into(), n, alpha.as_ptr() as *const _,
                         x.as_ptr() as *const _, incx, y.as_ptr() as *const _, incy,
                         ap.as_mut_ptr() as *mut _)
    }
}

#[inline]
/// matrix-matrix multiply (real single-precision)
/// 
/// SGEMM  performs one of the matrix-matrix operations
/// 
/// ```text
///    C := alpha*op( A )*op( B ) + beta*C,
/// ```
/// 
/// where  op( X ) is one of
/// 
/// ```text
///    op( X ) = X   or   op( X ) = X**T,
/// ```
/// 
/// `alpha` and `beta` are scalars, and `A`, `B` and `C` are matrices, with op( `A` )
/// an `m` by `k` matrix,  op( `B` )  a  `k` by `n` matrix and  `C` an `m` by `n` matrix.
pub fn sgemm(layout: Layout, transa: Transpose, transb: Transpose, m: i32, n: i32, k: i32,
             alpha: f32, a: &[f32], lda: i32, b: &[f32], ldb: i32, beta: f32, c: &mut [f32],
             ldc: i32) {

    unsafe {
        ffi::cblas_sgemm(layout.into(), transa.into(), transb.into(), m, n, k, alpha, a.as_ptr(),
                         lda, b.as_ptr(), ldb, beta, c.as_mut_ptr(), ldc)
    }
}

#[inline]
/// symmetric matrix-matrix multiply (real single-precision)
/// 
/// SSYMM  performs one of the matrix-matrix operations
/// 
/// ```text
///    C := alpha*A*B + beta*C,
/// ```
/// 
/// or
/// 
/// ```text
///    C := alpha*B*A + beta*C,
/// ```
/// 
/// where `alpha` and `beta` are scalars,  `A` is a symmetric matrix and  `B` and
/// `C` are  `m` by `n` matrices.
pub fn ssymm(layout: Layout, side: Side, uplo: Part, m: i32, n: i32, alpha: f32, a: &[f32],
             lda: i32, b: &[f32], ldb: i32, beta: f32, c: &mut [f32], ldc: i32) {

    unsafe {
        ffi::cblas_ssymm(layout.into(), side.into(), uplo.into(), m, n, alpha, a.as_ptr(), lda,
                         b.as_ptr(), ldb, beta, c.as_mut_ptr(), ldc)
    }
}

#[inline]
/// symmetric rank-k update to a matrix (real single-precision)
/// 
/// SSYRK  performs one of the symmetric rank `k` operations
/// 
/// ```text
///    C := alpha*A*A**T + beta*C,
/// ```
/// 
/// or
/// 
/// ```text
///    C := alpha*A**T*A + beta*C,
/// ```
/// 
/// where  `alpha` and `beta`  are scalars, `C` is an  `n` by `n`  symmetric matrix
/// and  `A`  is an  `n` by `k`  matrix in the first case and a  `k` by `n`  matrix
/// in the second case.
pub fn ssyrk(layout: Layout, uplo: Part, trans: Transpose, n: i32, k: i32, alpha: f32, a: &[f32],
             lda: i32, beta: f32, c: &mut [f32], ldc: i32) {

    unsafe {
        ffi::cblas_ssyrk(layout.into(), uplo.into(), trans.into(), n, k, alpha, a.as_ptr(), lda,
                         beta, c.as_mut_ptr(), ldc)
    }
}

#[inline]
/// symmetric rank-2k update to a matrix (real single-precision)
/// 
/// SSYR2K  performs one of the symmetric rank 2k operations
/// 
/// ```text
///    C := alpha*A*B**T + alpha*B*A**T + beta*C,
/// ```
/// 
/// or
/// 
/// ```text
///    C := alpha*A**T*B + alpha*B**T*A + beta*C,
/// ```
/// 
/// where  `alpha` and `beta`  are scalars, `C` is an  `n` by `n`  symmetric matrix
/// and  `A` and `B`  are  `n` by `k`  matrices  in the  first  case  and  `k` by `n`
/// matrices in the second case.
pub fn ssyr2k(layout: Layout, uplo: Part, trans: Transpose, n: i32, k: i32, alpha: f32, a: &[f32],
              lda: i32, b: &[f32], ldb: i32, beta: f32, c: &mut [f32], ldc: i32) {

    unsafe {
        ffi::cblas_ssyr2k(layout.into(), uplo.into(), trans.into(), n, k, alpha, a.as_ptr(), lda,
                          b.as_ptr(), ldb, beta, c.as_mut_ptr(), ldc)
    }
}

#[inline]
/// triangular matrix-matrix multiply (real single-precision)
/// 
/// STRMM  performs one of the matrix-matrix operations
/// 
/// ```text
///    B := alpha*op( A )*B,   or   B := alpha*B*op( A ),
/// ```
/// 
/// where  `alpha`  is a scalar,  `B`  is an `m` by `n` matrix,  `A`  is a unit, or
/// non-unit,  upper or lower triangular matrix  and  op( `A` )  is one  of
/// 
/// ```text
///    op( A ) = A   or   op( A ) = A**T.
/// ```
pub fn strmm(layout: Layout, side: Side, uplo: Part, transa: Transpose, diag: Diagonal, m: i32,
             n: i32, alpha: f32, a: &[f32], lda: i32, b: &mut [f32], ldb: i32) {

    unsafe {
        ffi::cblas_strmm(layout.into(), side.into(), uplo.into(), transa.into(), diag.into(), m, n,
                         alpha, a.as_ptr(), lda, b.as_mut_ptr(), ldb)
    }
}

#[inline]
/// solving triangular matrix with multiple right hand sides (real single-precision)
/// 
/// STRSM  solves one of the matrix equations
/// 
/// ```text
///    op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
/// ```
/// 
/// where `alpha` is a scalar, X and `B` are `m` by `n` matrices, `A` is a unit, or
/// non-unit,  upper or lower triangular matrix  and  op( `A` )  is one  of
/// 
/// ```text
///    op( A ) = A   or   op( A ) = A**T.
/// ```
/// 
/// The matrix X is overwritten on `B`.
pub fn strsm(layout: Layout, side: Side, uplo: Part, transa: Transpose, diag: Diagonal, m: i32,
             n: i32, alpha: f32, a: &[f32], lda: i32, b: &mut [f32], ldb: i32) {

    unsafe {
        ffi::cblas_strsm(layout.into(), side.into(), uplo.into(), transa.into(), diag.into(), m, n,
                         alpha, a.as_ptr(), lda, b.as_mut_ptr(), ldb)
    }
}

#[inline]
/// matrix-matrix multiply (real double-precision)
/// 
/// DGEMM  performs one of the matrix-matrix operations
/// 
/// ```text
///    C := alpha*op( A )*op( B ) + beta*C,
/// ```
/// 
/// where  op( X ) is one of
/// 
/// ```text
///    op( X ) = X   or   op( X ) = X**T,
/// ```
/// 
/// `alpha` and `beta` are scalars, and `A`, `B` and `C` are matrices, with op( `A` )
/// an `m` by `k` matrix,  op( `B` )  a  `k` by `n` matrix and  `C` an `m` by `n` matrix.
pub fn dgemm(layout: Layout, transa: Transpose, transb: Transpose, m: i32, n: i32, k: i32,
             alpha: f64, a: &[f64], lda: i32, b: &[f64], ldb: i32, beta: f64, c: &mut [f64],
             ldc: i32) {

    unsafe {
        ffi::cblas_dgemm(layout.into(), transa.into(), transb.into(), m, n, k, alpha, a.as_ptr(),
                         lda, b.as_ptr(), ldb, beta, c.as_mut_ptr(), ldc)
    }
}

#[inline]
/// symmetric matrix-matrix multiply (real double-precision)
/// 
/// DSYMM  performs one of the matrix-matrix operations
/// 
/// ```text
///    C := alpha*A*B + beta*C,
/// ```
/// 
/// or
/// 
/// ```text
///    C := alpha*B*A + beta*C,
/// ```
/// 
/// where `alpha` and `beta` are scalars,  `A` is a symmetric matrix and  `B` and
/// `C` are  `m` by `n` matrices.
pub fn dsymm(layout: Layout, side: Side, uplo: Part, m: i32, n: i32, alpha: f64, a: &[f64],
             lda: i32, b: &[f64], ldb: i32, beta: f64, c: &mut [f64], ldc: i32) {

    unsafe {
        ffi::cblas_dsymm(layout.into(), side.into(), uplo.into(), m, n, alpha, a.as_ptr(), lda,
                         b.as_ptr(), ldb, beta, c.as_mut_ptr(), ldc)
    }
}

#[inline]
/// symmetric rank-k update to a matrix (real double-precision)
/// 
/// DSYRK  performs one of the symmetric rank `k` operations
/// 
/// ```text
///    C := alpha*A*A**T + beta*C,
/// ```
/// 
/// or
/// 
/// ```text
///    C := alpha*A**T*A + beta*C,
/// ```
/// 
/// where  `alpha` and `beta`  are scalars, `C` is an  `n` by `n`  symmetric matrix
/// and  `A`  is an  `n` by `k`  matrix in the first case and a  `k` by `n`  matrix
/// in the second case.
pub fn dsyrk(layout: Layout, uplo: Part, trans: Transpose, n: i32, k: i32, alpha: f64, a: &[f64],
             lda: i32, beta: f64, c: &mut [f64], ldc: i32) {

    unsafe {
        ffi::cblas_dsyrk(layout.into(), uplo.into(), trans.into(), n, k, alpha, a.as_ptr(), lda,
                         beta, c.as_mut_ptr(), ldc)
    }
}

#[inline]
/// symmetric rank-2k update to a matrix (real double-precision)
/// 
/// DSYR2K  performs one of the symmetric rank 2k operations
/// 
/// ```text
///    C := alpha*A*B**T + alpha*B*A**T + beta*C,
/// ```
/// 
/// or
/// 
/// ```text
///    C := alpha*A**T*B + alpha*B**T*A + beta*C,
/// ```
/// 
/// where  `alpha` and `beta`  are scalars, `C` is an  `n` by `n`  symmetric matrix
/// and  `A` and `B`  are  `n` by `k`  matrices  in the  first  case  and  `k` by `n`
/// matrices in the second case.
pub fn dsyr2k(layout: Layout, uplo: Part, trans: Transpose, n: i32, k: i32, alpha: f64, a: &[f64],
              lda: i32, b: &[f64], ldb: i32, beta: f64, c: &mut [f64], ldc: i32) {

    unsafe {
        ffi::cblas_dsyr2k(layout.into(), uplo.into(), trans.into(), n, k, alpha, a.as_ptr(), lda,
                          b.as_ptr(), ldb, beta, c.as_mut_ptr(), ldc)
    }
}

#[inline]
/// triangular matrix-matrix multiply (real double-precision)
/// 
/// DTRMM  performs one of the matrix-matrix operations
/// 
/// ```text
///    B := alpha*op( A )*B,   or   B := alpha*B*op( A ),
/// ```
/// 
/// where  `alpha`  is a scalar,  `B`  is an `m` by `n` matrix,  `A`  is a unit, or
/// non-unit,  upper or lower triangular matrix  and  op( `A` )  is one  of
/// 
/// ```text
///    op( A ) = A   or   op( A ) = A**T.
/// ```
pub fn dtrmm(layout: Layout, side: Side, uplo: Part, transa: Transpose, diag: Diagonal, m: i32,
             n: i32, alpha: f64, a: &[f64], lda: i32, b: &mut [f64], ldb: i32) {

    unsafe {
        ffi::cblas_dtrmm(layout.into(), side.into(), uplo.into(), transa.into(), diag.into(), m, n,
                         alpha, a.as_ptr(), lda, b.as_mut_ptr(), ldb)
    }
}

#[inline]
/// solving triangular matrix with multiple right hand sides (real double-precision)
/// 
/// DTRSM  solves one of the matrix equations
/// 
/// ```text
///    op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
/// ```
/// 
/// where `alpha` is a scalar, X and `B` are `m` by `n` matrices, `A` is a unit, or
/// non-unit,  upper or lower triangular matrix  and  op( `A` )  is one  of
/// 
/// ```text
///    op( A ) = A   or   op( A ) = A**T.
/// ```
/// 
/// The matrix X is overwritten on `B`.
pub fn dtrsm(layout: Layout, side: Side, uplo: Part, transa: Transpose, diag: Diagonal, m: i32,
             n: i32, alpha: f64, a: &[f64], lda: i32, b: &mut [f64], ldb: i32) {

    unsafe {
        ffi::cblas_dtrsm(layout.into(), side.into(), uplo.into(), transa.into(), diag.into(), m, n,
                         alpha, a.as_ptr(), lda, b.as_mut_ptr(), ldb)
    }
}

#[inline]
/// matrix-matrix multiply (complex single-precision)
/// 
/// CGEMM  performs one of the matrix-matrix operations
/// 
/// ```text
///    C := alpha*op( A )*op( B ) + beta*C,
/// ```
/// 
/// where  op( X ) is one of
/// 
/// ```text
///    op( X ) = X   or   op( X ) = X**T   or   op( X ) = X**H,
/// ```
/// 
/// `alpha` and `beta` are scalars, and `A`, `B` and `C` are matrices, with op( `A` )
/// an `m` by `k` matrix,  op( `B` )  a  `k` by `n` matrix and  `C` an `m` by `n` matrix.
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
/// symmetric matrix-matrix multiply (complex single-precision)
/// 
/// CSYMM  performs one of the matrix-matrix operations
/// 
/// ```text
///    C := alpha*A*B + beta*C,
/// ```
/// 
/// or
/// 
/// ```text
///    C := alpha*B*A + beta*C,
/// ```
/// 
/// where  `alpha` and `beta` are scalars, `A` is a symmetric matrix and  `B` and
/// `C` are `m` by `n` matrices.
pub fn csymm(layout: Layout, side: Side, uplo: Part, m: i32, n: i32, alpha: &[c32], a: &[c32],
             lda: i32, b: &[c32], ldb: i32, beta: &[c32], c: &mut [c32], ldc: i32) {

    unsafe {
        ffi::cblas_csymm(layout.into(), side.into(), uplo.into(), m, n, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, b.as_ptr() as *const _, ldb,
                         beta.as_ptr() as *const _, c.as_mut_ptr() as *mut _, ldc)
    }
}

#[inline]
/// symmetric rank-k update to a matrix (complex single-precision)
/// 
/// CSYRK  performs one of the symmetric rank `k` operations
/// 
/// ```text
///    C := alpha*A*A**T + beta*C,
/// ```
/// 
/// or
/// 
/// ```text
///    C := alpha*A**T*A + beta*C,
/// ```
/// 
/// where  `alpha` and `beta`  are scalars,  `C` is an  `n` by `n` symmetric matrix
/// and  `A`  is an  `n` by `k`  matrix in the first case and a  `k` by `n`  matrix
/// in the second case.
pub fn csyrk(layout: Layout, uplo: Part, trans: Transpose, n: i32, k: i32, alpha: &[c32],
             a: &[c32], lda: i32, beta: &[c32], c: &mut [c32], ldc: i32) {

    unsafe {
        ffi::cblas_csyrk(layout.into(), uplo.into(), trans.into(), n, k,
                         alpha.as_ptr() as *const _, a.as_ptr() as *const _, lda,
                         beta.as_ptr() as *const _, c.as_mut_ptr() as *mut _, ldc)
    }
}

#[inline]
/// symmetric rank-2k update to a matrix (complex single-precision)
/// 
/// CSYR2K  performs one of the symmetric rank 2k operations
/// 
/// ```text
///    C := alpha*A*B**T + alpha*B*A**T + beta*C,
/// ```
/// 
/// or
/// 
/// ```text
///    C := alpha*A**T*B + alpha*B**T*A + beta*C,
/// ```
/// 
/// where  `alpha` and `beta`  are scalars,  `C` is an  `n` by `n` symmetric matrix
/// and  `A` and `B`  are  `n` by `k`  matrices  in the  first  case  and  `k` by `n`
/// matrices in the second case.
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
/// triangular matrix-matrix multiply (complex single-precision)
/// 
/// CTRMM  performs one of the matrix-matrix operations
/// 
/// ```text
///    B := alpha*op( A )*B,   or   B := alpha*B*op( A )
/// ```
/// 
/// where  `alpha`  is a scalar,  `B`  is an `m` by `n` matrix,  `A`  is a unit, or
/// non-unit,  upper or lower triangular matrix  and  op( `A` )  is one  of
/// 
/// ```text
///    op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H.
/// ```
pub fn ctrmm(layout: Layout, side: Side, uplo: Part, transa: Transpose, diag: Diagonal, m: i32,
             n: i32, alpha: &[c32], a: &[c32], lda: i32, b: &mut [c32], ldb: i32) {

    unsafe {
        ffi::cblas_ctrmm(layout.into(), side.into(), uplo.into(), transa.into(), diag.into(), m, n,
                         alpha.as_ptr() as *const _, a.as_ptr() as *const _, lda,
                         b.as_mut_ptr() as *mut _, ldb)
    }
}

#[inline]
/// solving triangular matrix with multiple right hand sides (complex single-precision)
/// 
/// CTRSM  solves one of the matrix equations
/// 
/// ```text
///    op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
/// ```
/// 
/// where `alpha` is a scalar, X and `B` are `m` by `n` matrices, `A` is a unit, or
/// non-unit,  upper or lower triangular matrix  and  op( `A` )  is one  of
/// 
/// ```text
///    op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H.
/// ```
/// 
/// The matrix X is overwritten on `B`.
pub fn ctrsm(layout: Layout, side: Side, uplo: Part, transa: Transpose, diag: Diagonal, m: i32,
             n: i32, alpha: &[c32], a: &[c32], lda: i32, b: &mut [c32], ldb: i32) {

    unsafe {
        ffi::cblas_ctrsm(layout.into(), side.into(), uplo.into(), transa.into(), diag.into(), m, n,
                         alpha.as_ptr() as *const _, a.as_ptr() as *const _, lda,
                         b.as_mut_ptr() as *mut _, ldb)
    }
}

#[inline]
/// matrix-matrix multiply (complex double-precision)
/// 
/// ZGEMM  performs one of the matrix-matrix operations
/// 
/// ```text
///    C := alpha*op( A )*op( B ) + beta*C,
/// ```
/// 
/// where  op( X ) is one of
/// 
/// ```text
///    op( X ) = X   or   op( X ) = X**T   or   op( X ) = X**H,
/// ```
/// 
/// `alpha` and `beta` are scalars, and `A`, `B` and `C` are matrices, with op( `A` )
/// an `m` by `k` matrix,  op( `B` )  a  `k` by `n` matrix and  `C` an `m` by `n` matrix.
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
/// symmetric matrix-matrix multiply (complex double-precision)
/// 
/// ZSYMM  performs one of the matrix-matrix operations
/// 
/// ```text
///    C := alpha*A*B + beta*C,
/// ```
/// 
/// or
/// 
/// ```text
///    C := alpha*B*A + beta*C,
/// ```
/// 
/// where  `alpha` and `beta` are scalars, `A` is a symmetric matrix and  `B` and
/// `C` are `m` by `n` matrices.
pub fn zsymm(layout: Layout, side: Side, uplo: Part, m: i32, n: i32, alpha: &[c64], a: &[c64],
             lda: i32, b: &[c64], ldb: i32, beta: &[c64], c: &mut [c64], ldc: i32) {

    unsafe {
        ffi::cblas_zsymm(layout.into(), side.into(), uplo.into(), m, n, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, b.as_ptr() as *const _, ldb,
                         beta.as_ptr() as *const _, c.as_mut_ptr() as *mut _, ldc)
    }
}

#[inline]
/// symmetric rank-k update to a matrix (complex double-precision)
/// 
/// ZSYRK  performs one of the symmetric rank `k` operations
/// 
/// ```text
///    C := alpha*A*A**T + beta*C,
/// ```
/// 
/// or
/// 
/// ```text
///    C := alpha*A**T*A + beta*C,
/// ```
/// 
/// where  `alpha` and `beta`  are scalars,  `C` is an  `n` by `n` symmetric matrix
/// and  `A`  is an  `n` by `k`  matrix in the first case and a  `k` by `n`  matrix
/// in the second case.
pub fn zsyrk(layout: Layout, uplo: Part, trans: Transpose, n: i32, k: i32, alpha: &[c64],
             a: &[c64], lda: i32, beta: &[c64], c: &mut [c64], ldc: i32) {

    unsafe {
        ffi::cblas_zsyrk(layout.into(), uplo.into(), trans.into(), n, k,
                         alpha.as_ptr() as *const _, a.as_ptr() as *const _, lda,
                         beta.as_ptr() as *const _, c.as_mut_ptr() as *mut _, ldc)
    }
}

#[inline]
/// symmetric rank-2k update to a matrix (complex double-precision)
/// 
/// ZSYR2K  performs one of the symmetric rank 2k operations
/// 
/// ```text
///    C := alpha*A*B**T + alpha*B*A**T + beta*C,
/// ```
/// 
/// or
/// 
/// ```text
///    C := alpha*A**T*B + alpha*B**T*A + beta*C,
/// ```
/// 
/// where  `alpha` and `beta`  are scalars,  `C` is an  `n` by `n` symmetric matrix
/// and  `A` and `B`  are  `n` by `k`  matrices  in the  first  case  and  `k` by `n`
/// matrices in the second case.
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
/// triangular matrix-matrix multiply (complex double-precision)
/// 
/// ZTRMM  performs one of the matrix-matrix operations
/// 
/// ```text
///    B := alpha*op( A )*B,   or   B := alpha*B*op( A )
/// ```
/// 
/// where  `alpha`  is a scalar,  `B`  is an `m` by `n` matrix,  `A`  is a unit, or
/// non-unit,  upper or lower triangular matrix  and  op( `A` )  is one  of
/// 
/// ```text
///    op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H.
/// ```
pub fn ztrmm(layout: Layout, side: Side, uplo: Part, transa: Transpose, diag: Diagonal, m: i32,
             n: i32, alpha: &[c64], a: &[c64], lda: i32, b: &mut [c64], ldb: i32) {

    unsafe {
        ffi::cblas_ztrmm(layout.into(), side.into(), uplo.into(), transa.into(), diag.into(), m, n,
                         alpha.as_ptr() as *const _, a.as_ptr() as *const _, lda,
                         b.as_mut_ptr() as *mut _, ldb)
    }
}

#[inline]
/// solving triangular matrix with multiple right hand sides (complex double-precision)
/// 
/// ZTRSM  solves one of the matrix equations
/// 
/// ```text
///    op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
/// ```
/// 
/// where `alpha` is a scalar, X and `B` are `m` by `n` matrices, `A` is a unit, or
/// non-unit,  upper or lower triangular matrix  and  op( `A` )  is one  of
/// 
/// ```text
///    op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H.
/// ```
/// 
/// The matrix X is overwritten on `B`.
pub fn ztrsm(layout: Layout, side: Side, uplo: Part, transa: Transpose, diag: Diagonal, m: i32,
             n: i32, alpha: &[c64], a: &[c64], lda: i32, b: &mut [c64], ldb: i32) {

    unsafe {
        ffi::cblas_ztrsm(layout.into(), side.into(), uplo.into(), transa.into(), diag.into(), m, n,
                         alpha.as_ptr() as *const _, a.as_ptr() as *const _, lda,
                         b.as_mut_ptr() as *mut _, ldb)
    }
}

#[inline]
/// hermitian matrix-matrix multiply (complex single-precision)
/// 
/// CHEMM  performs one of the matrix-matrix operations
/// 
/// ```text
///    C := alpha*A*B + beta*C,
/// ```
/// 
/// or
/// 
/// ```text
///    C := alpha*B*A + beta*C,
/// ```
/// 
/// where `alpha` and `beta` are scalars, `A` is an hermitian matrix and  `B` and
/// `C` are `m` by `n` matrices.
pub fn chemm(layout: Layout, side: Side, uplo: Part, m: i32, n: i32, alpha: &[c32], a: &[c32],
             lda: i32, b: &[c32], ldb: i32, beta: &[c32], c: &mut [c32], ldc: i32) {

    unsafe {
        ffi::cblas_chemm(layout.into(), side.into(), uplo.into(), m, n, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, b.as_ptr() as *const _, ldb,
                         beta.as_ptr() as *const _, c.as_mut_ptr() as *mut _, ldc)
    }
}

#[inline]
/// hermitian rank-k update to a matrix (complex single-precision)
/// 
/// CHERK  performs one of the hermitian rank `k` operations
/// 
/// ```text
///    C := alpha*A*A**H + beta*C,
/// ```
/// 
/// or
/// 
/// ```text
///    C := alpha*A**H*A + beta*C,
/// ```
/// 
/// where  `alpha` and `beta`  are  real scalars,  `C` is an  `n` by `n`  hermitian
/// matrix and  `A`  is an  `n` by `k`  matrix in the  first case and a  `k` by `n`
/// matrix in the second case.
pub fn cherk(layout: Layout, uplo: Part, trans: Transpose, n: i32, k: i32, alpha: f32, a: &[c32],
             lda: i32, beta: f32, c: &mut [c32], ldc: i32) {

    unsafe {
        ffi::cblas_cherk(layout.into(), uplo.into(), trans.into(), n, k, alpha,
                         a.as_ptr() as *const _, lda, beta, c.as_mut_ptr() as *mut _, ldc)
    }
}

#[inline]
/// hermitian rank-2k update to a matrix (complex single-precision)
/// 
/// CHER2K  performs one of the hermitian rank 2k operations
/// 
/// ```text
///    C := alpha*A*B**H + conjg( alpha )*B*A**H + beta*C,
/// ```
/// 
/// or
/// 
/// ```text
///    C := alpha*A**H*B + conjg( alpha )*B**H*A + beta*C,
/// ```
/// 
/// where  `alpha` and `beta`  are scalars with  `beta`  real,  `C` is an  `n` by `n`
/// hermitian matrix and  `A` and `B`  are  `n` by `k` matrices in the first case
/// and  `k` by `n`  matrices in the second case.
pub fn cher2k(layout: Layout, uplo: Part, trans: Transpose, n: i32, k: i32, alpha: &[c32],
              a: &[c32], lda: i32, b: &[c32], ldb: i32, beta: f32, c: &mut [c32], ldc: i32) {

    unsafe {
        ffi::cblas_cher2k(layout.into(), uplo.into(), trans.into(), n, k,
                          alpha.as_ptr() as *const _, a.as_ptr() as *const _, lda,
                          b.as_ptr() as *const _, ldb, beta, c.as_mut_ptr() as *mut _, ldc)
    }
}

#[inline]
/// hermitian matrix-matrix multiply (complex double-precision)
/// 
/// ZHEMM  performs one of the matrix-matrix operations
/// 
/// ```text
///    C := alpha*A*B + beta*C,
/// ```
/// 
/// or
/// 
/// ```text
///    C := alpha*B*A + beta*C,
/// ```
/// 
/// where `alpha` and `beta` are scalars, `A` is an hermitian matrix and  `B` and
/// `C` are `m` by `n` matrices.
pub fn zhemm(layout: Layout, side: Side, uplo: Part, m: i32, n: i32, alpha: &[c64], a: &[c64],
             lda: i32, b: &[c64], ldb: i32, beta: &[c64], c: &mut [c64], ldc: i32) {

    unsafe {
        ffi::cblas_zhemm(layout.into(), side.into(), uplo.into(), m, n, alpha.as_ptr() as *const _,
                         a.as_ptr() as *const _, lda, b.as_ptr() as *const _, ldb,
                         beta.as_ptr() as *const _, c.as_mut_ptr() as *mut _, ldc)
    }
}

#[inline]
/// hermitian rank-k update to a matrix (complex double-precision)
/// 
/// ZHERK  performs one of the hermitian rank `k` operations
/// 
/// ```text
///    C := alpha*A*A**H + beta*C,
/// ```
/// 
/// or
/// 
/// ```text
///    C := alpha*A**H*A + beta*C,
/// ```
/// 
/// where  `alpha` and `beta`  are  real scalars,  `C` is an  `n` by `n`  hermitian
/// matrix and  `A`  is an  `n` by `k`  matrix in the  first case and a  `k` by `n`
/// matrix in the second case.
pub fn zherk(layout: Layout, uplo: Part, trans: Transpose, n: i32, k: i32, alpha: f64, a: &[c64],
             lda: i32, beta: f64, c: &mut [c64], ldc: i32) {

    unsafe {
        ffi::cblas_zherk(layout.into(), uplo.into(), trans.into(), n, k, alpha,
                         a.as_ptr() as *const _, lda, beta, c.as_mut_ptr() as *mut _, ldc)
    }
}

#[inline]
/// hermitian rank-2k update to a matrix (complex double-precision)
/// 
/// ZHER2K  performs one of the hermitian rank 2k operations
/// 
/// ```text
///    C := alpha*A*B**H + conjg( alpha )*B*A**H + beta*C,
/// ```
/// 
/// or
/// 
/// ```text
///    C := alpha*A**H*B + conjg( alpha )*B**H*A + beta*C,
/// ```
/// 
/// where  `alpha` and `beta`  are scalars with  `beta`  real,  `C` is an  `n` by `n`
/// hermitian matrix and  `A` and `B`  are  `n` by `k` matrices in the first case
/// and  `k` by `n`  matrices in the second case.
pub fn zher2k(layout: Layout, uplo: Part, trans: Transpose, n: i32, k: i32, alpha: &[c64],
              a: &[c64], lda: i32, b: &[c64], ldb: i32, beta: f64, c: &mut [c64], ldc: i32) {

    unsafe {
        ffi::cblas_zher2k(layout.into(), uplo.into(), trans.into(), n, k,
                          alpha.as_ptr() as *const _, a.as_ptr() as *const _, lda,
                          b.as_ptr() as *const _, ldb, beta, c.as_mut_ptr() as *mut _, ldc)
    }
}

