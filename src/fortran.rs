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
/// setup Givens rotation (real single-precision)
/// 
/// SROTG construct givens plane rotation.
pub fn srotg(a: &mut f32, b: &mut f32, c: &mut f32, s: &mut f32) {
    unsafe {
        ffi::srotg_(a, b, c, s)
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
pub fn srotmg(d1: &mut f32, d2: &mut f32, x1: &mut f32, y1: f32, param: &mut [f32]) {
    unsafe {
        ffi::srotmg_(d1, d2, x1, &y1, param.as_mut_ptr())
    }
}

#[inline]
/// apply Givens rotation (real single-precision)
/// 
/// applies a plane rotation.
pub fn srot(n: i32, x: &mut [f32], incx: i32, y: &mut [f32], incy: i32, c: f32, s: f32) {
    unsafe {
        ffi::srot_(&n, x.as_mut_ptr(), &incx, y.as_mut_ptr(), &incy, &c, &s)
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
pub fn srotm(n: i32, x: &mut [f32], incx: i32, y: &mut [f32], incy: i32, param: &[f32]) {
    unsafe {
        ffi::srotm_(&n, x.as_mut_ptr(), &incx, y.as_mut_ptr(), &incy, param.as_ptr())
    }
}

#[inline]
/// swap x and y (real single-precision)
/// 
/// interchanges two vectors.
/// uses unrolled loops for increments equal to 1.
pub fn sswap(n: i32, x: &mut [f32], incx: i32, y: &mut [f32], incy: i32) {
    unsafe {
        ffi::sswap_(&n, x.as_mut_ptr(), &incx, y.as_mut_ptr(), &incy)
    }
}

#[inline]
/// x = alpha*x (real single-precision)
/// 
/// scales a vector by a constant.
/// uses unrolled loops for increment equal to 1.
pub fn sscal(n: i32, a: f32, x: &mut [f32], incx: i32) {
    unsafe {
        ffi::sscal_(&n, &a, x.as_mut_ptr(), &incx)
    }
}

#[inline]
/// copy x into y (real single-precision)
/// 
/// SCOPY copies a vector, `x`, to a vector, `y`.
/// uses unrolled loops for increments equal to 1.
pub fn scopy(n: i32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
    unsafe {
        ffi::scopy_(&n, x.as_ptr(), &incx, y.as_mut_ptr(), &incy)
    }
}

#[inline]
/// y = alpha*x + y (real single-precision)
/// 
/// SAXPY constant times a vector plus a vector.
/// uses unrolled loops for increments equal to one.
pub fn saxpy(n: i32, alpha: f32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
    unsafe {
        ffi::saxpy_(&n, &alpha, x.as_ptr(), &incx, y.as_mut_ptr(), &incy)
    }
}

#[inline]
/// dot product (real single-precision)
/// 
/// SDOT forms the dot product of two vectors.
/// uses unrolled loops for increments equal to one.
pub fn sdot(n: i32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
    unsafe {
        ffi::sdot_(&n, x.as_ptr(), &incx, y.as_ptr(), &incy)
    }
}

#[inline]
/// dot product with extended precision accumulation (real single-precision)
/// 
///
pub fn sdsdot(n: i32, sb: &[f32], x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
    unsafe {
        ffi::sdsdot_(&n, sb.as_ptr(), x.as_ptr(), &incx, y.as_ptr(), &incy)
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
        ffi::snrm2_(&n, x.as_ptr(), &incx)
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
        ffi::scnrm2_(&n, x.as_ptr() as *const _, &incx)
    }
}

#[inline]
/// sum of absolute values (real single-precision)
/// 
/// SASUM takes the sum of the absolute values.
/// uses unrolled loops for increment equal to one.
pub fn sasum(n: i32, x: &[f32], incx: i32) -> f32 {
    unsafe {
        ffi::sasum_(&n, x.as_ptr(), &incx)
    }
}

#[inline]
/// index of max abs value (real single-precision)
/// 
/// ISAMAX finds the index of the first element having maximum absolute value.
pub fn isamax(n: i32, x: &[f32], incx: i32) -> usize {
    unsafe {
        ffi::isamax_(&n, x.as_ptr(), &incx) as usize
    }
}

#[inline]
/// setup Givens rotation (real double-precision)
/// 
/// DROTG construct givens plane rotation.
pub fn drotg(a: &mut f64, b: &mut f64, c: &mut f64, s: &mut f64) {
    unsafe {
        ffi::drotg_(a, b, c, s)
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
pub fn drotmg(d1: &mut f64, d2: &mut f64, x1: &mut f64, y1: f64, param: &mut [f64]) {
    unsafe {
        ffi::drotmg_(d1, d2, x1, &y1, param.as_mut_ptr())
    }
}

#[inline]
/// apply Givens rotation (real double-precision)
/// 
/// DROT applies a plane rotation.
pub fn drot(n: i32, x: &mut [f64], incx: i32, y: &mut [f64], incy: i32, c: f64, s: f64) {
    unsafe {
        ffi::drot_(&n, x.as_mut_ptr(), &incx, y.as_mut_ptr(), &incy, &c, &s)
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
pub fn drotm(n: i32, x: &mut [f64], incx: i32, y: &mut [f64], incy: i32, param: &[f64]) {
    unsafe {
        ffi::drotm_(&n, x.as_mut_ptr(), &incx, y.as_mut_ptr(), &incy, param.as_ptr())
    }
}

#[inline]
/// swap x and y (real double-precision)
/// 
/// interchanges two vectors.
/// uses unrolled loops for increments equal one.
pub fn dswap(n: i32, x: &mut [f64], incx: i32, y: &mut [f64], incy: i32) {
    unsafe {
        ffi::dswap_(&n, x.as_mut_ptr(), &incx, y.as_mut_ptr(), &incy)
    }
}

#[inline]
/// x = alpha*x (real double-precision)
/// 
/// DSCAL scales a vector by a constant.
/// uses unrolled loops for increment equal to one.
pub fn dscal(n: i32, a: f64, x: &mut [f64], incx: i32) {
    unsafe {
        ffi::dscal_(&n, &a, x.as_mut_ptr(), &incx)
    }
}

#[inline]
/// copy x into y (real double-precision)
/// 
/// DCOPY copies a vector, `x`, to a vector, `y`.
/// uses unrolled loops for increments equal to one.
pub fn dcopy(n: i32, x: &[f64], incx: i32, y: &mut [f64], incy: i32) {
    unsafe {
        ffi::dcopy_(&n, x.as_ptr(), &incx, y.as_mut_ptr(), &incy)
    }
}

#[inline]
/// y = alpha*x + y (real double-precision)
/// 
/// DAXPY constant times a vector plus a vector.
/// uses unrolled loops for increments equal to one.
pub fn daxpy(n: i32, alpha: f64, x: &[f64], incx: i32, y: &mut [f64], incy: i32) {
    unsafe {
        ffi::daxpy_(&n, &alpha, x.as_ptr(), &incx, y.as_mut_ptr(), &incy)
    }
}

#[inline]
/// dot product (real double-precision)
/// 
/// DDOT forms the dot product of two vectors.
/// uses unrolled loops for increments equal to one.
pub fn ddot(n: i32, x: &[f64], incx: i32, y: &[f64], incy: i32) -> f64 {
    unsafe {
        ffi::ddot_(&n, x.as_ptr(), &incx, y.as_ptr(), &incy)
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
        ffi::dsdot_(&n, x.as_ptr(), &incx, y.as_ptr(), &incy)
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
        ffi::dnrm2_(&n, x.as_ptr(), &incx)
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
        ffi::dznrm2_(&n, x.as_ptr() as *const _, &incx)
    }
}

#[inline]
/// sum of absolute values (real double-precision)
/// 
/// DASUM takes the sum of the absolute values.
pub fn dasum(n: i32, x: &[f64], incx: i32) -> f64 {
    unsafe {
        ffi::dasum_(&n, x.as_ptr(), &incx)
    }
}

#[inline]
/// index of max abs value (real double-precision)
/// 
/// IDAMAX finds the index of the first element having maximum absolute value.
pub fn idamax(n: i32, x: &[f64], incx: i32) -> usize {
    unsafe {
        ffi::idamax_(&n, x.as_ptr(), &incx) as usize
    }
}

#[inline]
/// setup Givens rotation (complex single-precision)
/// 
/// CROTG determines a complex Givens rotation.
pub fn crotg(a: &mut c32, b: c32, c: &mut f32, s: &mut c32) {
    unsafe {
        ffi::crotg_(a as *mut _ as *mut _, &b as *const _ as *const _, c, s as *mut _ as *mut _)
    }
}

#[inline]
/// apply Givens rotation (complex single-precision)
/// 
/// CSROT applies a plane rotation, where the cos and sin (c and `s`) are real
/// and the vectors cx and cy are complex.
/// jack dongarra, linpack, 3/11/78.
pub fn csrot(n: i32, x: &mut [c32], incx: i32, y: &mut [c32], incy: i32, c: f32, s: f32) {
    unsafe {
        ffi::csrot_(&n, x.as_mut_ptr() as *mut _, &incx, y.as_mut_ptr() as *mut _, &incy, &c, &s)
    }
}

#[inline]
/// swap x and y (complex single-precision)
/// 
///   CSWAP interchanges two vectors.
pub fn cswap(n: i32, x: &mut [c32], incx: i32, y: &mut [c32], incy: i32) {
    unsafe {
        ffi::cswap_(&n, x.as_mut_ptr() as *mut _, &incx, y.as_mut_ptr() as *mut _, &incy)
    }
}

#[inline]
/// x = alpha*x (complex single-precision)
/// 
/// CSCAL scales a vector by a constant.
pub fn cscal(n: i32, a: c32, x: &mut [c32], incx: i32) {
    unsafe {
        ffi::cscal_(&n, &a as *const _ as *const _, x.as_mut_ptr() as *mut _, &incx)
    }
}

#[inline]
/// x = alpha*x, scalar alpha (complex single-precision)
/// 
/// CSSCAL scales a complex vector by a real constant.
pub fn csscal(n: i32, a: f32, x: &mut [c32], incx: i32) {
    unsafe {
        ffi::csscal_(&n, &a, x.as_mut_ptr() as *mut _, &incx)
    }
}

#[inline]
/// copy x into y (complex single-precision)
/// 
/// CCOPY copies a vector `x` to a vector `y`.
pub fn ccopy(n: i32, x: &[c32], incx: i32, y: &mut [c32], incy: i32) {
    unsafe {
        ffi::ccopy_(&n, x.as_ptr() as *const _, &incx, y.as_mut_ptr() as *mut _, &incy)
    }
}

#[inline]
/// y = alpha*x + y (complex single-precision)
/// 
/// CAXPY constant times a vector plus a vector.
pub fn caxpy(n: i32, alpha: c32, x: &[c32], incx: i32, y: &mut [c32], incy: i32) {
    unsafe {
        ffi::caxpy_(&n, &alpha as *const _ as *const _, x.as_ptr() as *const _, &incx,
                    y.as_mut_ptr() as *mut _, &incy)
    }
}

#[inline]
/// y = x<sup>T</sup> * y, dot product (complex single-precision)
/// 
/// CDOTU forms the dot product of two complex vectors
/// ```text
///      CDOTU = X^T * Y
/// ```
pub fn cdotu(pres: &mut [c32], n: i32, x: &[c32], incx: i32, y: &[c32], incy: i32) {
    unsafe {
        ffi::cdotu_(pres.as_mut_ptr() as *mut _, &n, x.as_ptr() as *const _, &incx,
                    y.as_ptr() as *const _, &incy)
    }
}

#[inline]
/// y = x<sup>T</sup> * y, dot product with first argument conjugated (complex single-precision)
/// 
/// CDOTC forms the dot product of two complex vectors
/// ```text
///      CDOTC = X^H * Y
/// ```
pub fn cdotc(pres: &mut [c32], n: i32, x: &[c32], incx: i32, y: &[c32], incy: i32) {
    unsafe {
        ffi::cdotc_(pres.as_mut_ptr() as *mut _, &n, x.as_ptr() as *const _, &incx,
                    y.as_ptr() as *const _, &incy)
    }
}

#[inline]
/// sum of absolute values (single-precision)
/// 
/// SCASUM takes the sum of the (|Re(.)| + |Im(.)|)'s of a complex vector and
/// returns a single precision result.
pub fn scasum(n: i32, x: &[c32], incx: i32) -> f32 {
    unsafe {
        ffi::scasum_(&n, x.as_ptr() as *const _, &incx)
    }
}

#[inline]
/// index of max abs value (complex single-precision)
/// 
/// ICAMAX finds the index of the first element having maximum |Re(.)| + |Im(.)|
pub fn icamax(n: i32, x: &[c32], incx: i32) -> usize {
    unsafe {
        ffi::icamax_(&n, x.as_ptr() as *const _, &incx) as usize
    }
}

#[inline]
/// setup Givens rotation (complex double-precision)
/// 
/// ZROTG determines a double complex Givens rotation.
pub fn zrotg(a: &mut c64, b: c64, c: &mut f64, s: &mut c64) {
    unsafe {
        ffi::zrotg_(a as *mut _ as *mut _, &b as *const _ as *const _, c, s as *mut _ as *mut _)
    }
}

#[inline]
/// apply Givens rotation (complex double-precision)
/// 
/// Applies a plane rotation, where the cos and sin (c and `s`) are real
/// and the vectors cx and cy are complex.
/// jack dongarra, linpack, 3/11/78.
pub fn zdrot(n: i32, x: &mut [c64], incx: i32, y: &mut [c64], incy: i32, c: f64, s: f64) {
    unsafe {
        ffi::zdrot_(&n, x.as_mut_ptr() as *mut _, &incx, y.as_mut_ptr() as *mut _, &incy, &c, &s)
    }
}

#[inline]
/// swap x and y (complex double-precision)
/// 
/// ZSWAP interchanges two vectors.
pub fn zswap(n: i32, x: &mut [c64], incx: i32, y: &mut [c64], incy: i32) {
    unsafe {
        ffi::zswap_(&n, x.as_mut_ptr() as *mut _, &incx, y.as_mut_ptr() as *mut _, &incy)
    }
}

#[inline]
/// x = alpha*x (complex double-precision)
/// 
/// ZSCAL scales a vector by a constant.
pub fn zscal(n: i32, a: c64, x: &mut [c64], incx: i32) {
    unsafe {
        ffi::zscal_(&n, &a as *const _ as *const _, x.as_mut_ptr() as *mut _, &incx)
    }
}

#[inline]
/// x = alpha*x, scalar alpha (complex double-precision)
/// 
/// ZDSCAL scales a vector by a constant.
pub fn zdscal(n: i32, a: f64, x: &mut [c64], incx: i32) {
    unsafe {
        ffi::zdscal_(&n, &a, x.as_mut_ptr() as *mut _, &incx)
    }
}

#[inline]
/// copy x into y (complex double-precision)
/// 
/// ZCOPY copies a vector, `x`, to a vector, `y`.
pub fn zcopy(n: i32, x: &[c64], incx: i32, y: &mut [c64], incy: i32) {
    unsafe {
        ffi::zcopy_(&n, x.as_ptr() as *const _, &incx, y.as_mut_ptr() as *mut _, &incy)
    }
}

#[inline]
/// y = alpha*x + y (complex double-precision)
/// 
/// ZAXPY constant times a vector plus a vector.
pub fn zaxpy(n: i32, alpha: c64, x: &[c64], incx: i32, y: &mut [c64], incy: i32) {
    unsafe {
        ffi::zaxpy_(&n, &alpha as *const _ as *const _, x.as_ptr() as *const _, &incx,
                    y.as_mut_ptr() as *mut _, &incy)
    }
}

#[inline]
/// y = x<sup>T</sup> * y, dot product (complex double-precision)
/// 
/// ZDOTU forms the dot product of two complex vectors
/// ```text
///      ZDOTU = X^T * Y
/// ```
pub fn zdotu(pres: &mut [c64], n: i32, x: &[c64], incx: i32, y: &[c64], incy: i32) {
    unsafe {
        ffi::zdotu_(pres.as_mut_ptr() as *mut _, &n, x.as_ptr() as *const _, &incx,
                    y.as_ptr() as *const _, &incy)
    }
}

#[inline]
/// y = x<sup>T</sup> * y, dot product with first argument conjugated (complex double-precision)
/// 
/// ZDOTC forms the dot product of two complex vectors
/// ```text
///      ZDOTC = X^H * Y
/// ```
pub fn zdotc(pres: &mut [c64], n: i32, x: &[c64], incx: i32, y: &[c64], incy: i32) {
    unsafe {
        ffi::zdotc_(pres.as_mut_ptr() as *mut _, &n, x.as_ptr() as *const _, &incx,
                    y.as_ptr() as *const _, &incy)
    }
}

#[inline]
/// sum of absolute values (double-precision)
/// 
/// DZASUM takes the sum of the (|Re(.)| + |Im(.)|)'s of a complex vector and
/// returns a single precision result.
pub fn dzasum(n: i32, x: &[c64], incx: i32) -> f64 {
    unsafe {
        ffi::dzasum_(&n, x.as_ptr() as *const _, &incx)
    }
}

#[inline]
/// index of max abs value (complex double-precision)
/// 
/// IZAMAX finds the index of the first element having maximum |Re(.)| + |Im(.)|
pub fn izamax(n: i32, x: &[c64], incx: i32) -> usize {
    unsafe {
        ffi::izamax_(&n, x.as_ptr() as *const _, &incx) as usize
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
pub fn sgemv(trans: u8, m: i32, n: i32, alpha: f32, a: &[f32], lda: i32, x: &[f32], incx: i32,
             beta: f32, y: &mut [f32], incy: i32) {

    unsafe {
        ffi::sgemv_(&(trans as c_char), &m, &n, &alpha, a.as_ptr(), &lda, x.as_ptr(), &incx, &beta,
                    y.as_mut_ptr(), &incy)
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
pub fn sgbmv(trans: u8, m: i32, n: i32, kl: i32, ku: i32, alpha: f32, a: &[f32], lda: i32,
             x: &[f32], incx: i32, beta: f32, y: &mut [f32], incy: i32) {

    unsafe {
        ffi::sgbmv_(&(trans as c_char), &m, &n, &kl, &ku, &alpha, a.as_ptr(), &lda, x.as_ptr(),
                    &incx, &beta, y.as_mut_ptr(), &incy)
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
pub fn ssymv(uplo: u8, n: i32, alpha: f32, a: &[f32], lda: i32, x: &[f32], incx: i32, beta: f32,
             y: &mut [f32], incy: i32) {

    unsafe {
        ffi::ssymv_(&(uplo as c_char), &n, &alpha, a.as_ptr(), &lda, x.as_ptr(), &incx, &beta,
                    y.as_mut_ptr(), &incy)
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
pub fn ssbmv(uplo: u8, n: i32, k: i32, alpha: f32, a: &[f32], lda: i32, x: &[f32], incx: i32,
             beta: f32, y: &mut [f32], incy: i32) {

    unsafe {
        ffi::ssbmv_(&(uplo as c_char), &n, &k, &alpha, a.as_ptr(), &lda, x.as_ptr(), &incx, &beta,
                    y.as_mut_ptr(), &incy)
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
pub fn sspmv(uplo: u8, n: i32, alpha: f32, ap: &[f32], x: &[f32], incx: i32, beta: f32,
             y: &mut [f32], incy: i32) {

    unsafe {
        ffi::sspmv_(&(uplo as c_char), &n, &alpha, ap.as_ptr(), x.as_ptr(), &incx, &beta,
                    y.as_mut_ptr(), &incy)
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
/// where x is an `n` element vector and  `A` is an `n` by `n` unit, or non-unit,
/// upper or lower triangular matrix.
pub fn strmv(uplo: u8, transa: u8, diag: u8, n: i32, a: &[f32], lda: i32, b: &mut [f32],
             incx: i32) {

    unsafe {
        ffi::strmv_(&(uplo as c_char), &(transa as c_char), &(diag as c_char), &n, a.as_ptr(),
                    &lda, b.as_mut_ptr(), &incx)
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
pub fn stbmv(uplo: u8, trans: u8, diag: u8, n: i32, k: i32, a: &[f32], lda: i32, x: &mut [f32],
             incx: i32) {

    unsafe {
        ffi::stbmv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &n, &k, a.as_ptr(),
                    &lda, x.as_mut_ptr(), &incx)
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
pub fn stpmv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[f32], x: &mut [f32], incx: i32) {
    unsafe {
        ffi::stpmv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &n, ap.as_ptr(),
                    x.as_mut_ptr(), &incx)
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
pub fn strsv(uplo: u8, trans: u8, diag: u8, n: i32, a: &[f32], lda: i32, x: &mut [f32],
             incx: i32) {

    unsafe {
        ffi::strsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &n, a.as_ptr(), &lda,
                    x.as_mut_ptr(), &incx)
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
pub fn stbsv(uplo: u8, trans: u8, diag: u8, n: i32, k: i32, a: &[f32], lda: i32, x: &mut [f32],
             incx: i32) {

    unsafe {
        ffi::stbsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &n, &k, a.as_ptr(),
                    &lda, x.as_mut_ptr(), &incx)
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
pub fn stpsv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[f32], x: &mut [f32], incx: i32) {
    unsafe {
        ffi::stpsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &n, ap.as_ptr(),
                    x.as_mut_ptr(), &incx)
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
pub fn sger(m: i32, n: i32, alpha: f32, x: &[f32], incx: i32, y: &[f32], incy: i32, a: &mut [f32],
            lda: i32) {

    unsafe {
        ffi::sger_(&m, &n, &alpha, x.as_ptr(), &incx, y.as_ptr(), &incy, a.as_mut_ptr(), &lda)
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
pub fn ssyr(uplo: u8, n: i32, alpha: f32, x: &[f32], incx: i32, a: &mut [f32], lda: i32) {
    unsafe {
        ffi::ssyr_(&(uplo as c_char), &n, &alpha, x.as_ptr(), &incx, a.as_mut_ptr(), &lda)
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
pub fn sspr(uplo: u8, n: i32, alpha: f32, x: &[f32], incx: i32, ap: &mut [f32]) {
    unsafe {
        ffi::sspr_(&(uplo as c_char), &n, &alpha, x.as_ptr(), &incx, ap.as_mut_ptr())
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
pub fn ssyr2(uplo: u8, n: i32, alpha: f32, x: &[f32], incx: i32, y: &[f32], incy: i32,
             a: &mut [f32], lda: i32) {

    unsafe {
        ffi::ssyr2_(&(uplo as c_char), &n, &alpha, x.as_ptr(), &incx, y.as_ptr(), &incy,
                    a.as_mut_ptr(), &lda)
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
/// where `alpha` is a scalar, `x` and `y` are `n` element vectors and A is an
/// `n` by `n` symmetric matrix, supplied in packed form.
pub fn sspr2(uplo: u8, n: i32, alpha: f32, x: &[f32], incx: i32, y: &[f32], incy: i32,
             ap: &mut [f32]) {

    unsafe {
        ffi::sspr2_(&(uplo as c_char), &n, &alpha, x.as_ptr(), &incx, y.as_ptr(), &incy,
                    ap.as_mut_ptr())
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
pub fn dgemv(trans: u8, m: i32, n: i32, alpha: f64, a: &[f64], lda: i32, x: &[f64], incx: i32,
             beta: f64, y: &mut [f64], incy: i32) {

    unsafe {
        ffi::dgemv_(&(trans as c_char), &m, &n, &alpha, a.as_ptr(), &lda, x.as_ptr(), &incx, &beta,
                    y.as_mut_ptr(), &incy)
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
pub fn dgbmv(trans: u8, m: i32, n: i32, kl: i32, ku: i32, alpha: f64, a: &[f64], lda: i32,
             x: &[f64], incx: i32, beta: f64, y: &mut [f64], incy: i32) {

    unsafe {
        ffi::dgbmv_(&(trans as c_char), &m, &n, &kl, &ku, &alpha, a.as_ptr(), &lda, x.as_ptr(),
                    &incx, &beta, y.as_mut_ptr(), &incy)
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
pub fn dsymv(uplo: u8, n: i32, alpha: f64, a: &[f64], lda: i32, x: &[f64], incx: i32, beta: f64,
             y: &mut [f64], incy: i32) {

    unsafe {
        ffi::dsymv_(&(uplo as c_char), &n, &alpha, a.as_ptr(), &lda, x.as_ptr(), &incx, &beta,
                    y.as_mut_ptr(), &incy)
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
pub fn dsbmv(uplo: u8, n: i32, k: i32, alpha: f64, a: &[f64], lda: i32, x: &[f64], incx: i32,
             beta: f64, y: &mut [f64], incy: i32) {

    unsafe {
        ffi::dsbmv_(&(uplo as c_char), &n, &k, &alpha, a.as_ptr(), &lda, x.as_ptr(), &incx, &beta,
                    y.as_mut_ptr(), &incy)
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
pub fn dspmv(uplo: u8, n: i32, alpha: f64, ap: &[f64], x: &[f64], incx: i32, beta: f64,
             y: &mut [f64], incy: i32) {

    unsafe {
        ffi::dspmv_(&(uplo as c_char), &n, &alpha, ap.as_ptr(), x.as_ptr(), &incx, &beta,
                    y.as_mut_ptr(), &incy)
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
/// where x is an `n` element vector and  `A` is an `n` by `n` unit, or non-unit,
/// upper or lower triangular matrix.
pub fn dtrmv(uplo: u8, transa: u8, diag: u8, n: i32, a: &[f64], lda: i32, b: &mut [f64],
             incx: i32) {

    unsafe {
        ffi::dtrmv_(&(uplo as c_char), &(transa as c_char), &(diag as c_char), &n, a.as_ptr(),
                    &lda, b.as_mut_ptr(), &incx)
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
pub fn dtbmv(uplo: u8, trans: u8, diag: u8, n: i32, k: i32, a: &[f64], lda: i32, x: &mut [f64],
             incx: i32) {

    unsafe {
        ffi::dtbmv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &n, &k, a.as_ptr(),
                    &lda, x.as_mut_ptr(), &incx)
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
pub fn dtpmv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[f64], x: &mut [f64], incx: i32) {
    unsafe {
        ffi::dtpmv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &n, ap.as_ptr(),
                    x.as_mut_ptr(), &incx)
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
pub fn dtrsv(uplo: u8, trans: u8, diag: u8, n: i32, a: &[f64], lda: i32, x: &mut [f64],
             incx: i32) {

    unsafe {
        ffi::dtrsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &n, a.as_ptr(), &lda,
                    x.as_mut_ptr(), &incx)
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
pub fn dtbsv(uplo: u8, trans: u8, diag: u8, n: i32, k: i32, a: &[f64], lda: i32, x: &mut [f64],
             incx: i32) {

    unsafe {
        ffi::dtbsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &n, &k, a.as_ptr(),
                    &lda, x.as_mut_ptr(), &incx)
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
pub fn dtpsv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[f64], x: &mut [f64], incx: i32) {
    unsafe {
        ffi::dtpsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &n, ap.as_ptr(),
                    x.as_mut_ptr(), &incx)
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
pub fn dger(m: i32, n: i32, alpha: f64, x: &[f64], incx: i32, y: &[f64], incy: i32, a: &mut [f64],
            lda: i32) {

    unsafe {
        ffi::dger_(&m, &n, &alpha, x.as_ptr(), &incx, y.as_ptr(), &incy, a.as_mut_ptr(), &lda)
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
pub fn dsyr(uplo: u8, n: i32, alpha: f64, x: &[f64], incx: i32, a: &mut [f64], lda: i32) {
    unsafe {
        ffi::dsyr_(&(uplo as c_char), &n, &alpha, x.as_ptr(), &incx, a.as_mut_ptr(), &lda)
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
pub fn dspr(uplo: u8, n: i32, alpha: f64, x: &[f64], incx: i32, ap: &mut [f64]) {
    unsafe {
        ffi::dspr_(&(uplo as c_char), &n, &alpha, x.as_ptr(), &incx, ap.as_mut_ptr())
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
pub fn dsyr2(uplo: u8, n: i32, alpha: f64, x: &[f64], incx: i32, y: &[f64], incy: i32,
             a: &mut [f64], lda: i32) {

    unsafe {
        ffi::dsyr2_(&(uplo as c_char), &n, &alpha, x.as_ptr(), &incx, y.as_ptr(), &incy,
                    a.as_mut_ptr(), &lda)
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
/// where `alpha` is a scalar, `x` and `y` are `n` element vectors and A is an
/// `n` by `n` symmetric matrix, supplied in packed form.
pub fn dspr2(uplo: u8, n: i32, alpha: f64, x: &[f64], incx: i32, y: &[f64], incy: i32,
             ap: &mut [f64]) {

    unsafe {
        ffi::dspr2_(&(uplo as c_char), &n, &alpha, x.as_ptr(), &incx, y.as_ptr(), &incy,
                    ap.as_mut_ptr())
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
pub fn cgemv(trans: u8, m: i32, n: i32, alpha: c32, a: &[c32], lda: i32, x: &[c32], incx: i32,
             beta: c32, y: &mut [c32], incy: i32) {

    unsafe {
        ffi::cgemv_(&(trans as c_char), &m, &n, &alpha as *const _ as *const _,
                    a.as_ptr() as *const _, &lda, x.as_ptr() as *const _, &incx,
                    &beta as *const _ as *const _, y.as_mut_ptr() as *mut _, &incy)
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
pub fn cgbmv(trans: u8, m: i32, n: i32, kl: i32, ku: i32, alpha: c32, a: &[c32], lda: i32,
             x: &[c32], incx: i32, beta: c32, y: &mut [c32], incy: i32) {

    unsafe {
        ffi::cgbmv_(&(trans as c_char), &m, &n, &kl, &ku, &alpha as *const _ as *const _,
                    a.as_ptr() as *const _, &lda, x.as_ptr() as *const _, &incx,
                    &beta as *const _ as *const _, y.as_mut_ptr() as *mut _, &incy)
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
pub fn chemv(uplo: u8, n: i32, alpha: c32, a: &[c32], lda: i32, x: &[c32], incx: i32, beta: c32,
             y: &mut [c32], incy: i32) {

    unsafe {
        ffi::chemv_(&(uplo as c_char), &n, &alpha as *const _ as *const _, a.as_ptr() as *const _,
                    &lda, x.as_ptr() as *const _, &incx, &beta as *const _ as *const _,
                    y.as_mut_ptr() as *mut _, &incy)
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
pub fn chbmv(uplo: u8, n: i32, k: i32, alpha: c32, a: &[c32], lda: i32, x: &[c32], incx: i32,
             beta: c32, y: &mut [c32], incy: i32) {

    unsafe {
        ffi::chbmv_(&(uplo as c_char), &n, &k, &alpha as *const _ as *const _,
                    a.as_ptr() as *const _, &lda, x.as_ptr() as *const _, &incx,
                    &beta as *const _ as *const _, y.as_mut_ptr() as *mut _, &incy)
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
pub fn chpmv(uplo: u8, n: i32, alpha: c32, ap: &[c32], x: &[c32], incx: i32, beta: c32,
             y: &mut [c32], incy: i32) {

    unsafe {
        ffi::chpmv_(&(uplo as c_char), &n, &alpha as *const _ as *const _, ap.as_ptr() as *const _,
                    x.as_ptr() as *const _, &incx, &beta as *const _ as *const _,
                    y.as_mut_ptr() as *mut _, &incy)
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
/// where x is an `n` element vector and  `A` is an `n` by `n` unit, or non-unit,
/// upper or lower triangular matrix.
pub fn ctrmv(uplo: u8, transa: u8, diag: u8, n: i32, a: &[c32], lda: i32, b: &mut [c32],
             incx: i32) {

    unsafe {
        ffi::ctrmv_(&(uplo as c_char), &(transa as c_char), &(diag as c_char), &n,
                    a.as_ptr() as *const _, &lda, b.as_mut_ptr() as *mut _, &incx)
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
pub fn ctbmv(uplo: u8, trans: u8, diag: u8, n: i32, k: i32, a: &[c32], lda: i32, x: &mut [c32],
             incx: i32) {

    unsafe {
        ffi::ctbmv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &n, &k,
                    a.as_ptr() as *const _, &lda, x.as_mut_ptr() as *mut _, &incx)
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
pub fn ctpmv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[c32], x: &mut [c32], incx: i32) {
    unsafe {
        ffi::ctpmv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &n,
                    ap.as_ptr() as *const _, x.as_mut_ptr() as *mut _, &incx)
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
pub fn ctrsv(uplo: u8, trans: u8, diag: u8, n: i32, a: &[c32], lda: i32, x: &mut [c32],
             incx: i32) {

    unsafe {
        ffi::ctrsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &n,
                    a.as_ptr() as *const _, &lda, x.as_mut_ptr() as *mut _, &incx)
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
pub fn ctbsv(uplo: u8, trans: u8, diag: u8, n: i32, k: i32, a: &[c32], lda: i32, x: &mut [c32],
             incx: i32) {

    unsafe {
        ffi::ctbsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &n, &k,
                    a.as_ptr() as *const _, &lda, x.as_mut_ptr() as *mut _, &incx)
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
pub fn ctpsv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[c32], x: &mut [c32], incx: i32) {
    unsafe {
        ffi::ctpsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &n,
                    ap.as_ptr() as *const _, x.as_mut_ptr() as *mut _, &incx)
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
pub fn cgeru(m: i32, n: i32, alpha: c32, x: &[c32], incx: i32, y: &[c32], incy: i32, a: &mut [c32],
             lda: i32) {

    unsafe {
        ffi::cgeru_(&m, &n, &alpha as *const _ as *const _, x.as_ptr() as *const _, &incx,
                    y.as_ptr() as *const _, &incy, a.as_mut_ptr() as *mut _, &lda)
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
pub fn cgerc(m: i32, n: i32, alpha: c32, x: &[c32], incx: i32, y: &[c32], incy: i32, a: &mut [c32],
             lda: i32) {

    unsafe {
        ffi::cgerc_(&m, &n, &alpha as *const _ as *const _, x.as_ptr() as *const _, &incx,
                    y.as_ptr() as *const _, &incy, a.as_mut_ptr() as *mut _, &lda)
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
pub fn cher(uplo: u8, n: i32, alpha: f32, x: &[c32], incx: i32, a: &mut [c32], lda: i32) {
    unsafe {
        ffi::cher_(&(uplo as c_char), &n, &alpha, x.as_ptr() as *const _, &incx,
                   a.as_mut_ptr() as *mut _, &lda)
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
/// where `alpha` is a real scalar, `x` is an `n` element vector and A is an
/// `n` by `n` hermitian matrix, supplied in packed form.
pub fn chpr(uplo: u8, n: i32, alpha: f32, x: &[c32], incx: i32, ap: &mut [c32]) {
    unsafe {
        ffi::chpr_(&(uplo as c_char), &n, &alpha, x.as_ptr() as *const _, &incx,
                   ap.as_mut_ptr() as *mut _)
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
pub fn chpr2(uplo: u8, n: i32, alpha: c32, x: &[c32], incx: i32, y: &[c32], incy: i32,
             ap: &mut [c32]) {

    unsafe {
        ffi::chpr2_(&(uplo as c_char), &n, &alpha as *const _ as *const _, x.as_ptr() as *const _,
                    &incx, y.as_ptr() as *const _, &incy, ap.as_mut_ptr() as *mut _)
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
pub fn cher2(uplo: u8, n: i32, alpha: c32, x: &[c32], incx: i32, y: &[c32], incy: i32,
             a: &mut [c32], lda: i32) {

    unsafe {
        ffi::cher2_(&(uplo as c_char), &n, &alpha as *const _ as *const _, x.as_ptr() as *const _,
                    &incx, y.as_ptr() as *const _, &incy, a.as_mut_ptr() as *mut _, &lda)
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
pub fn zgemv(trans: u8, m: i32, n: i32, alpha: c64, a: &[c64], lda: i32, x: &[c64], incx: i32,
             beta: c64, y: &mut [c64], incy: i32) {

    unsafe {
        ffi::zgemv_(&(trans as c_char), &m, &n, &alpha as *const _ as *const _,
                    a.as_ptr() as *const _, &lda, x.as_ptr() as *const _, &incx,
                    &beta as *const _ as *const _, y.as_mut_ptr() as *mut _, &incy)
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
pub fn zgbmv(trans: u8, m: i32, n: i32, kl: i32, ku: i32, alpha: c64, a: &[c64], lda: i32,
             x: &[c64], incx: i32, beta: c64, y: &mut [c64], incy: i32) {

    unsafe {
        ffi::zgbmv_(&(trans as c_char), &m, &n, &kl, &ku, &alpha as *const _ as *const _,
                    a.as_ptr() as *const _, &lda, x.as_ptr() as *const _, &incx,
                    &beta as *const _ as *const _, y.as_mut_ptr() as *mut _, &incy)
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
pub fn zhemv(uplo: u8, n: i32, alpha: c64, a: &[c64], lda: i32, x: &[c64], incx: i32, beta: c64,
             y: &mut [c64], incy: i32) {

    unsafe {
        ffi::zhemv_(&(uplo as c_char), &n, &alpha as *const _ as *const _, a.as_ptr() as *const _,
                    &lda, x.as_ptr() as *const _, &incx, &beta as *const _ as *const _,
                    y.as_mut_ptr() as *mut _, &incy)
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
pub fn zhbmv(uplo: u8, n: i32, k: i32, alpha: c64, a: &[c64], lda: i32, x: &[c64], incx: i32,
             beta: c64, y: &mut [c64], incy: i32) {

    unsafe {
        ffi::zhbmv_(&(uplo as c_char), &n, &k, &alpha as *const _ as *const _,
                    a.as_ptr() as *const _, &lda, x.as_ptr() as *const _, &incx,
                    &beta as *const _ as *const _, y.as_mut_ptr() as *mut _, &incy)
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
pub fn zhpmv(uplo: u8, n: i32, alpha: c64, ap: &[c64], x: &[c64], incx: i32, beta: c64,
             y: &mut [c64], incy: i32) {

    unsafe {
        ffi::zhpmv_(&(uplo as c_char), &n, &alpha as *const _ as *const _, ap.as_ptr() as *const _,
                    x.as_ptr() as *const _, &incx, &beta as *const _ as *const _,
                    y.as_mut_ptr() as *mut _, &incy)
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
/// where x is an `n` element vector and  `A` is an `n` by `n` unit, or non-unit,
/// upper or lower triangular matrix.
pub fn ztrmv(uplo: u8, transa: u8, diag: u8, n: i32, a: &[c64], lda: i32, b: &mut [c64],
             incx: i32) {

    unsafe {
        ffi::ztrmv_(&(uplo as c_char), &(transa as c_char), &(diag as c_char), &n,
                    a.as_ptr() as *const _, &lda, b.as_mut_ptr() as *mut _, &incx)
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
pub fn ztbmv(uplo: u8, trans: u8, diag: u8, n: i32, k: i32, a: &[c64], lda: i32, x: &mut [c64],
             incx: i32) {

    unsafe {
        ffi::ztbmv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &n, &k,
                    a.as_ptr() as *const _, &lda, x.as_mut_ptr() as *mut _, &incx)
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
pub fn ztpmv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[c64], x: &mut [c64], incx: i32) {
    unsafe {
        ffi::ztpmv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &n,
                    ap.as_ptr() as *const _, x.as_mut_ptr() as *mut _, &incx)
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
pub fn ztrsv(uplo: u8, trans: u8, diag: u8, n: i32, a: &[c64], lda: i32, x: &mut [c64],
             incx: i32) {

    unsafe {
        ffi::ztrsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &n,
                    a.as_ptr() as *const _, &lda, x.as_mut_ptr() as *mut _, &incx)
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
pub fn ztbsv(uplo: u8, trans: u8, diag: u8, n: i32, k: i32, a: &[c64], lda: i32, x: &mut [c64],
             incx: i32) {

    unsafe {
        ffi::ztbsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &n, &k,
                    a.as_ptr() as *const _, &lda, x.as_mut_ptr() as *mut _, &incx)
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
pub fn ztpsv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[c64], x: &mut [c64], incx: i32) {
    unsafe {
        ffi::ztpsv_(&(uplo as c_char), &(trans as c_char), &(diag as c_char), &n,
                    ap.as_ptr() as *const _, x.as_mut_ptr() as *mut _, &incx)
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
pub fn zgeru(m: i32, n: i32, alpha: c64, x: &[c64], incx: i32, y: &[c64], incy: i32, a: &mut [c64],
             lda: i32) {

    unsafe {
        ffi::zgeru_(&m, &n, &alpha as *const _ as *const _, x.as_ptr() as *const _, &incx,
                    y.as_ptr() as *const _, &incy, a.as_mut_ptr() as *mut _, &lda)
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
pub fn zgerc(m: i32, n: i32, alpha: c64, x: &[c64], incx: i32, y: &[c64], incy: i32, a: &mut [c64],
             lda: i32) {

    unsafe {
        ffi::zgerc_(&m, &n, &alpha as *const _ as *const _, x.as_ptr() as *const _, &incx,
                    y.as_ptr() as *const _, &incy, a.as_mut_ptr() as *mut _, &lda)
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
pub fn zher(uplo: u8, n: i32, alpha: f64, x: &[c64], incx: i32, a: &mut [c64], lda: i32) {
    unsafe {
        ffi::zher_(&(uplo as c_char), &n, &alpha, x.as_ptr() as *const _, &incx,
                   a.as_mut_ptr() as *mut _, &lda)
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
/// where `alpha` is a real scalar, `x` is an `n` element vector and A is an
/// `n` by `n` hermitian matrix, supplied in packed form.
pub fn zhpr(uplo: u8, n: i32, alpha: f64, x: &[c64], incx: i32, ap: &mut [c64]) {
    unsafe {
        ffi::zhpr_(&(uplo as c_char), &n, &alpha, x.as_ptr() as *const _, &incx,
                   ap.as_mut_ptr() as *mut _)
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
pub fn zher2(uplo: u8, n: i32, alpha: c64, x: &[c64], incx: i32, y: &[c64], incy: i32,
             a: &mut [c64], lda: i32) {

    unsafe {
        ffi::zher2_(&(uplo as c_char), &n, &alpha as *const _ as *const _, x.as_ptr() as *const _,
                    &incx, y.as_ptr() as *const _, &incy, a.as_mut_ptr() as *mut _, &lda)
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
pub fn zhpr2(uplo: u8, n: i32, alpha: c64, x: &[c64], incx: i32, y: &[c64], incy: i32,
             ap: &mut [c64]) {

    unsafe {
        ffi::zhpr2_(&(uplo as c_char), &n, &alpha as *const _ as *const _, x.as_ptr() as *const _,
                    &incx, y.as_ptr() as *const _, &incy, ap.as_mut_ptr() as *mut _)
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
pub fn sgemm(transa: u8, transb: u8, m: i32, n: i32, k: i32, alpha: f32, a: &[f32], lda: i32,
             b: &[f32], ldb: i32, beta: f32, c: &mut [f32], ldc: i32) {

    unsafe {
        ffi::sgemm_(&(transa as c_char), &(transb as c_char), &m, &n, &k, &alpha, a.as_ptr(), &lda,
                    b.as_ptr(), &ldb, &beta, c.as_mut_ptr(), &ldc)
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
pub fn ssymm(side: u8, uplo: u8, m: i32, n: i32, alpha: f32, a: &[f32], lda: i32, b: &[f32],
             ldb: i32, beta: f32, c: &mut [f32], ldc: i32) {

    unsafe {
        ffi::ssymm_(&(side as c_char), &(uplo as c_char), &m, &n, &alpha, a.as_ptr(), &lda,
                    b.as_ptr(), &ldb, &beta, c.as_mut_ptr(), &ldc)
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
pub fn ssyrk(uplo: u8, trans: u8, n: i32, k: i32, alpha: f32, a: &[f32], lda: i32, beta: f32,
             c: &mut [f32], ldc: i32) {

    unsafe {
        ffi::ssyrk_(&(uplo as c_char), &(trans as c_char), &n, &k, &alpha, a.as_ptr(), &lda, &beta,
                    c.as_mut_ptr(), &ldc)
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
pub fn ssyr2k(uplo: u8, trans: u8, n: i32, k: i32, alpha: f32, a: &[f32], lda: i32, b: &[f32],
              ldb: i32, beta: f32, c: &mut [f32], ldc: i32) {

    unsafe {
        ffi::ssyr2k_(&(uplo as c_char), &(trans as c_char), &n, &k, &alpha, a.as_ptr(), &lda,
                     b.as_ptr(), &ldb, &beta, c.as_mut_ptr(), &ldc)
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
pub fn strmm(side: u8, uplo: u8, transa: u8, diag: u8, m: i32, n: i32, alpha: f32, a: &[f32],
             lda: i32, b: &mut [f32], ldb: i32) {

    unsafe {
        ffi::strmm_(&(side as c_char), &(uplo as c_char), &(transa as c_char), &(diag as c_char),
                    &m, &n, &alpha, a.as_ptr(), &lda, b.as_mut_ptr(), &ldb)
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
pub fn strsm(side: u8, uplo: u8, transa: u8, diag: u8, m: i32, n: i32, alpha: f32, a: &[f32],
             lda: i32, b: &mut [f32], ldb: i32) {

    unsafe {
        ffi::strsm_(&(side as c_char), &(uplo as c_char), &(transa as c_char), &(diag as c_char),
                    &m, &n, &alpha, a.as_ptr(), &lda, b.as_mut_ptr(), &ldb)
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
pub fn dgemm(transa: u8, transb: u8, m: i32, n: i32, k: i32, alpha: f64, a: &[f64], lda: i32,
             b: &[f64], ldb: i32, beta: f64, c: &mut [f64], ldc: i32) {

    unsafe {
        ffi::dgemm_(&(transa as c_char), &(transb as c_char), &m, &n, &k, &alpha, a.as_ptr(), &lda,
                    b.as_ptr(), &ldb, &beta, c.as_mut_ptr(), &ldc)
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
pub fn dsymm(side: u8, uplo: u8, m: i32, n: i32, alpha: f64, a: &[f64], lda: i32, b: &[f64],
             ldb: i32, beta: f64, c: &mut [f64], ldc: i32) {

    unsafe {
        ffi::dsymm_(&(side as c_char), &(uplo as c_char), &m, &n, &alpha, a.as_ptr(), &lda,
                    b.as_ptr(), &ldb, &beta, c.as_mut_ptr(), &ldc)
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
pub fn dsyrk(uplo: u8, trans: u8, n: i32, k: i32, alpha: f64, a: &[f64], lda: i32, beta: f64,
             c: &mut [f64], ldc: i32) {

    unsafe {
        ffi::dsyrk_(&(uplo as c_char), &(trans as c_char), &n, &k, &alpha, a.as_ptr(), &lda, &beta,
                    c.as_mut_ptr(), &ldc)
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
pub fn dsyr2k(uplo: u8, trans: u8, n: i32, k: i32, alpha: f64, a: &[f64], lda: i32, b: &[f64],
              ldb: i32, beta: f64, c: &mut [f64], ldc: i32) {

    unsafe {
        ffi::dsyr2k_(&(uplo as c_char), &(trans as c_char), &n, &k, &alpha, a.as_ptr(), &lda,
                     b.as_ptr(), &ldb, &beta, c.as_mut_ptr(), &ldc)
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
pub fn dtrmm(side: u8, uplo: u8, transa: u8, diag: u8, m: i32, n: i32, alpha: f64, a: &[f64],
             lda: i32, b: &mut [f64], ldb: i32) {

    unsafe {
        ffi::dtrmm_(&(side as c_char), &(uplo as c_char), &(transa as c_char), &(diag as c_char),
                    &m, &n, &alpha, a.as_ptr(), &lda, b.as_mut_ptr(), &ldb)
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
pub fn dtrsm(side: u8, uplo: u8, transa: u8, diag: u8, m: i32, n: i32, alpha: f64, a: &[f64],
             lda: i32, b: &mut [f64], ldb: i32) {

    unsafe {
        ffi::dtrsm_(&(side as c_char), &(uplo as c_char), &(transa as c_char), &(diag as c_char),
                    &m, &n, &alpha, a.as_ptr(), &lda, b.as_mut_ptr(), &ldb)
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
pub fn cgemm(transa: u8, transb: u8, m: i32, n: i32, k: i32, alpha: c32, a: &[c32], lda: i32,
             b: &[c32], ldb: i32, beta: c32, c: &mut [c32], ldc: i32) {

    unsafe {
        ffi::cgemm_(&(transa as c_char), &(transb as c_char), &m, &n, &k,
                    &alpha as *const _ as *const _, a.as_ptr() as *const _, &lda,
                    b.as_ptr() as *const _, &ldb, &beta as *const _ as *const _,
                    c.as_mut_ptr() as *mut _, &ldc)
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
pub fn csymm(side: u8, uplo: u8, m: i32, n: i32, alpha: c32, a: &[c32], lda: i32, b: &[c32],
             ldb: i32, beta: c32, c: &mut [c32], ldc: i32) {

    unsafe {
        ffi::csymm_(&(side as c_char), &(uplo as c_char), &m, &n, &alpha as *const _ as *const _,
                    a.as_ptr() as *const _, &lda, b.as_ptr() as *const _, &ldb,
                    &beta as *const _ as *const _, c.as_mut_ptr() as *mut _, &ldc)
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
pub fn chemm(side: u8, uplo: u8, m: i32, n: i32, alpha: c32, a: &[c32], lda: i32, b: &[c32],
             ldb: i32, beta: c32, c: &mut [c32], ldc: i32) {

    unsafe {
        ffi::chemm_(&(side as c_char), &(uplo as c_char), &m, &n, &alpha as *const _ as *const _,
                    a.as_ptr() as *const _, &lda, b.as_ptr() as *const _, &ldb,
                    &beta as *const _ as *const _, c.as_mut_ptr() as *mut _, &ldc)
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
pub fn csyrk(uplo: u8, trans: u8, n: i32, k: i32, alpha: c32, a: &[c32], lda: i32, beta: c32,
             c: &mut [c32], ldc: i32) {

    unsafe {
        ffi::csyrk_(&(uplo as c_char), &(trans as c_char), &n, &k, &alpha as *const _ as *const _,
                    a.as_ptr() as *const _, &lda, &beta as *const _ as *const _,
                    c.as_mut_ptr() as *mut _, &ldc)
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
pub fn cherk(uplo: u8, trans: u8, n: i32, k: i32, alpha: f32, a: &[c32], lda: i32, beta: f32,
             c: &mut [c32], ldc: i32) {

    unsafe {
        ffi::cherk_(&(uplo as c_char), &(trans as c_char), &n, &k, &alpha, a.as_ptr() as *const _,
                    &lda, &beta, c.as_mut_ptr() as *mut _, &ldc)
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
pub fn csyr2k(uplo: u8, trans: u8, n: i32, k: i32, alpha: c32, a: &[c32], lda: i32, b: &[c32],
              ldb: i32, beta: c32, c: &mut [c32], ldc: i32) {

    unsafe {
        ffi::csyr2k_(&(uplo as c_char), &(trans as c_char), &n, &k, &alpha as *const _ as *const _,
                     a.as_ptr() as *const _, &lda, b.as_ptr() as *const _, &ldb,
                     &beta as *const _ as *const _, c.as_mut_ptr() as *mut _, &ldc)
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
pub fn cher2k(uplo: u8, trans: u8, n: i32, k: i32, alpha: c32, a: &[c32], lda: i32, b: &[c32],
              ldb: i32, beta: f32, c: &mut [c32], ldc: i32) {

    unsafe {
        ffi::cher2k_(&(uplo as c_char), &(trans as c_char), &n, &k, &alpha as *const _ as *const _,
                     a.as_ptr() as *const _, &lda, b.as_ptr() as *const _, &ldb, &beta,
                     c.as_mut_ptr() as *mut _, &ldc)
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
pub fn ctrmm(side: u8, uplo: u8, transa: u8, diag: u8, m: i32, n: i32, alpha: c32, a: &[c32],
             lda: i32, b: &mut [c32], ldb: i32) {

    unsafe {
        ffi::ctrmm_(&(side as c_char), &(uplo as c_char), &(transa as c_char), &(diag as c_char),
                    &m, &n, &alpha as *const _ as *const _, a.as_ptr() as *const _, &lda,
                    b.as_mut_ptr() as *mut _, &ldb)
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
pub fn ctrsm(side: u8, uplo: u8, transa: u8, diag: u8, m: i32, n: i32, alpha: c32, a: &[c32],
             lda: i32, b: &mut [c32], ldb: i32) {

    unsafe {
        ffi::ctrsm_(&(side as c_char), &(uplo as c_char), &(transa as c_char), &(diag as c_char),
                    &m, &n, &alpha as *const _ as *const _, a.as_ptr() as *const _, &lda,
                    b.as_mut_ptr() as *mut _, &ldb)
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
pub fn zgemm(transa: u8, transb: u8, m: i32, n: i32, k: i32, alpha: c64, a: &[c64], lda: i32,
             b: &[c64], ldb: i32, beta: c64, c: &mut [c64], ldc: i32) {

    unsafe {
        ffi::zgemm_(&(transa as c_char), &(transb as c_char), &m, &n, &k,
                    &alpha as *const _ as *const _, a.as_ptr() as *const _, &lda,
                    b.as_ptr() as *const _, &ldb, &beta as *const _ as *const _,
                    c.as_mut_ptr() as *mut _, &ldc)
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
pub fn zsymm(side: u8, uplo: u8, m: i32, n: i32, alpha: c64, a: &[c64], lda: i32, b: &[c64],
             ldb: i32, beta: c64, c: &mut [c64], ldc: i32) {

    unsafe {
        ffi::zsymm_(&(side as c_char), &(uplo as c_char), &m, &n, &alpha as *const _ as *const _,
                    a.as_ptr() as *const _, &lda, b.as_ptr() as *const _, &ldb,
                    &beta as *const _ as *const _, c.as_mut_ptr() as *mut _, &ldc)
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
pub fn zhemm(side: u8, uplo: u8, m: i32, n: i32, alpha: c64, a: &[c64], lda: i32, b: &[c64],
             ldb: i32, beta: c64, c: &mut [c64], ldc: i32) {

    unsafe {
        ffi::zhemm_(&(side as c_char), &(uplo as c_char), &m, &n, &alpha as *const _ as *const _,
                    a.as_ptr() as *const _, &lda, b.as_ptr() as *const _, &ldb,
                    &beta as *const _ as *const _, c.as_mut_ptr() as *mut _, &ldc)
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
pub fn zsyrk(uplo: u8, trans: u8, n: i32, k: i32, alpha: c64, a: &[c64], lda: i32, beta: c64,
             c: &mut [c64], ldc: i32) {

    unsafe {
        ffi::zsyrk_(&(uplo as c_char), &(trans as c_char), &n, &k, &alpha as *const _ as *const _,
                    a.as_ptr() as *const _, &lda, &beta as *const _ as *const _,
                    c.as_mut_ptr() as *mut _, &ldc)
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
pub fn zherk(uplo: u8, trans: u8, n: i32, k: i32, alpha: f64, a: &[c64], lda: i32, beta: f64,
             c: &mut [c64], ldc: i32) {

    unsafe {
        ffi::zherk_(&(uplo as c_char), &(trans as c_char), &n, &k, &alpha, a.as_ptr() as *const _,
                    &lda, &beta, c.as_mut_ptr() as *mut _, &ldc)
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
pub fn zsyr2k(uplo: u8, trans: u8, n: i32, k: i32, alpha: c64, a: &[c64], lda: i32, b: &[c64],
              ldb: i32, beta: c64, c: &mut [c64], ldc: i32) {

    unsafe {
        ffi::zsyr2k_(&(uplo as c_char), &(trans as c_char), &n, &k, &alpha as *const _ as *const _,
                     a.as_ptr() as *const _, &lda, b.as_ptr() as *const _, &ldb,
                     &beta as *const _ as *const _, c.as_mut_ptr() as *mut _, &ldc)
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
pub fn zher2k(uplo: u8, trans: u8, n: i32, k: i32, alpha: c64, a: &[c64], lda: i32, b: &[c64],
              ldb: i32, beta: f64, c: &mut [c64], ldc: i32) {

    unsafe {
        ffi::zher2k_(&(uplo as c_char), &(trans as c_char), &n, &k, &alpha as *const _ as *const _,
                     a.as_ptr() as *const _, &lda, b.as_ptr() as *const _, &ldb, &beta,
                     c.as_mut_ptr() as *mut _, &ldc)
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
pub fn ztrmm(side: u8, uplo: u8, transa: u8, diag: u8, m: i32, n: i32, alpha: c64, a: &[c64],
             lda: i32, b: &mut [c64], ldb: i32) {

    unsafe {
        ffi::ztrmm_(&(side as c_char), &(uplo as c_char), &(transa as c_char), &(diag as c_char),
                    &m, &n, &alpha as *const _ as *const _, a.as_ptr() as *const _, &lda,
                    b.as_mut_ptr() as *mut _, &ldb)
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
pub fn ztrsm(side: u8, uplo: u8, transa: u8, diag: u8, m: i32, n: i32, alpha: c64, a: &[c64],
             lda: i32, b: &mut [c64], ldb: i32) {

    unsafe {
        ffi::ztrsm_(&(side as c_char), &(uplo as c_char), &(transa as c_char), &(diag as c_char),
                    &m, &n, &alpha as *const _ as *const _, a.as_ptr() as *const _, &lda,
                    b.as_mut_ptr() as *mut _, &ldb)
    }
}

