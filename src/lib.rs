//! An interface to the [Basic Linear Algebra Subprograms][1].
//!
//! [1]: http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms

#![cfg_attr(test, feature(test))]

#[cfg(test)]
#[macro_use]
extern crate assert;

#[cfg(test)]
extern crate test;

extern crate num;
extern crate libc;

extern crate "libblas-sys" as raw;

type C = num::Complex<f32>;
type Z = num::Complex<f64>;

use raw::*;
use libc::size_t;
use std::mem::transmute;
use std::cmp::min;

pub enum Layout {
    RowMajor = CblasRowMajor as isize,
    ColumnMajor = CblasColMajor as isize,
}

pub enum Transpose {
    None = CblasNoTrans as isize,
    Transpose = CblasTrans as isize,
    ConjugateTranspose = CblasConjTrans as isize,
}

#[inline]
pub fn dgemv(layout: Layout, trans: Transpose, m: usize, n: usize, alpha: f64,
             a: &[f64], lda: usize, x: &[f64], incx: usize, beta: f64,
             y: &mut [f64], incy: usize) {

    unsafe {
        cblas_dgemv(layout as u32, trans as u32, m as i32, n as i32, alpha,
                         SliceExt::as_ptr(a), lda as i32, SliceExt::as_ptr(x), incx as i32, beta,
                         SliceExt::as_mut_ptr(y), incy as i32);
    }
}

#[inline]
pub fn dgemm(layout: Layout, transa: Transpose, transb: Transpose, m: usize,
             n: usize, k: usize, alpha: f64, a: &[f64], lda: usize, b: &[f64],
             ldb: usize, beta: f64, c: &mut [f64], ldc: usize) {

    unsafe {
        cblas_dgemm(layout as u32, transa as u32, transb as u32,
                         m as i32, n as i32, k as i32, alpha, SliceExt::as_ptr(a),
                         lda as i32, SliceExt::as_ptr(b), ldb as i32, beta,
                         SliceExt::as_mut_ptr(c), ldc as i32);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn dgemv() {
        let (m, n) = (2, 3);

        let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![6.0, 8.0];

        ::dgemv(::Layout::ColumnMajor, ::Transpose::None,
                m, n, 1.0, &a, m, &x, 1, 1.0, &mut y, 1);

        let expected_y = vec![20.0, 40.0];
        assert_equal!(y, expected_y);
    }

    #[test]
    fn dgemm() {
        let (m, n, k) = (2, 4, 3);

        let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let b = vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0];
        let mut c = vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0];

        ::dgemm(::Layout::ColumnMajor, ::Transpose::None, ::Transpose::None,
                m, n, k, 1.0, &a, m, &b, k, 1.0, &mut c, m);

        let expected_c = vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0];
        assert_equal!(c, expected_c);
    }
}

/// Trait for all types that BLAS supports: Float, double, float complex, double complex.
///
/// You may be wondering, "What is that mysterious Float type, and why don't you just use Self?"
/// Well, consider what happens if you want to implement this trait for `Complex<f32>`. Returning
/// Self isn't always the right thing, sometimes you want to return the f32 directly. There's no
/// other good way to handle that.
///
/// So, in many cases, `Self` is used. `Float` is used when the function directly references the
/// inner float type representation for. `RetSelf` is used to paper over the difference between
/// functions returning `_Complex` and `Self` being a Rust type. `Weird` is used similarly, but for
/// scalar arguments that need a pointer to `Float` for Complex types.
pub unsafe trait Num: Copy {
    type Float: Copy;
    type RetSelf: Copy;
    type Weird: Copy;

    fn dot() -> unsafe extern fn(n: blasint, x: *const Self::Float, incx: blasint, y: *const Self::Float, incy: blasint) -> Self::RetSelf;
    fn axpy() -> unsafe extern fn(n: blasint, alpha: Self, x: *const Self, incx: blasint, y: *mut Self, incy: blasint);
    fn axpby() -> unsafe extern fn(n: blasint, alpha: Self, x: *const Self, incx: blasint, beta: Self, y: *mut Self, incy: blasint);
    fn rot() -> unsafe extern fn(N: blasint, X: *mut Self, incX: blasint, Y: *mut Self, incY: blasint, c: Self::Float, s: Self);
    fn rotg() -> unsafe extern fn(a: *mut Self::Float, b: *mut Self::Float, c: *mut Self::Float, s: *mut Self);
    fn scal() -> unsafe extern fn(N: blasint, alpha: Self, X: *mut Self, incX: blasint);
    fn asum() -> unsafe extern fn(N: blasint, x: *const Self::Float, incx: blasint) -> Self::Float;
    fn iamax() -> unsafe extern fn(N: blasint, x: *const Self::Float, incx: blasint) -> size_t;
    fn nrm2() -> unsafe extern fn(N: blasint, X: *const Self::Float, incX: blasint) -> Self::Float;

    fn gemv() -> unsafe extern fn(order: CBLAS_ORDER, trans: CBLAS_TRANSPOSE, m: blasint, n: blasint, alpha: <Self as Num>::Weird, a: *const <Self as Num>::Float, lda: blasint, x: *const <Self as Num>::Float, incx: blasint, beta: <Self as Num>::Weird, y: *mut <Self as Num>::Float, incy: blasint) -> ();
    fn ger() -> unsafe extern fn(order: CBLAS_ORDER, M: blasint, N: blasint, alpha: <Self as Num>::Weird, X: *const <Self as Num>::Float, incX: blasint, Y: *const <Self as Num>::Float, incY: blasint, A: *mut <Self as Num>::Float, lda: blasint) -> ();
    fn trsv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, A: *const <Self as Num>::Float, lda: blasint, X: *mut <Self as Num>::Float, incX: blasint) -> ();
    fn trmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, A: *const <Self as Num>::Float, lda: blasint, X: *mut <Self as Num>::Float, incX: blasint) -> ();
    fn gbmv() -> unsafe extern fn(order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, M: blasint, N: blasint, KL: blasint, KU: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, X: *const <Self as Num>::Float, incX: blasint, beta: <Self as Num>::Weird, Y: *mut <Self as Num>::Float, incY: blasint) -> ();
    fn tbmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, K: blasint, A: *const <Self as Num>::Float, lda: blasint, X: *mut <Self as Num>::Float, incX: blasint) -> ();
    fn tpmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, Ap: *const <Self as Num>::Float, X: *mut <Self as Num>::Float, incX: blasint) -> ();
    fn tbsv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, K: blasint, A: *const <Self as Num>::Float, lda: blasint, X: *mut <Self as Num>::Float, incX: blasint) -> ();
    fn tpsv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, Ap: *const <Self as Num>::Float, X: *mut <Self as Num>::Float, incX: blasint) -> ();

    fn gemm() -> unsafe extern fn(Order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, TransB: CBLAS_TRANSPOSE, M: blasint, N: blasint, K: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, B: *const <Self as Num>::Float, ldb: blasint, beta: <Self as Num>::Weird, C: *mut <Self as Num>::Float, ldc: blasint) -> ();
    fn symm() -> unsafe extern fn(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, M: blasint, N: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, B: *const <Self as Num>::Float, ldb: blasint, beta: <Self as Num>::Weird, C: *mut <Self as Num>::Float, ldc: blasint) -> ();
    fn syrk() -> unsafe extern fn(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, beta: <Self as Num>::Weird, C: *mut <Self as Num>::Float, ldc: blasint) -> ();
    fn syr2k() -> unsafe extern fn(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, B: *const <Self as Num>::Float, ldb: blasint, beta: <Self as Num>::Weird, C: *mut <Self as Num>::Float, ldc: blasint) -> ();
    fn trmm() -> unsafe extern fn(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, M: blasint, N: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, B: *mut <Self as Num>::Float, ldb: blasint) -> ();
    fn trsm() -> unsafe extern fn(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, M: blasint, N: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, B: *mut <Self as Num>::Float, ldb: blasint) -> ();
}

/// A trait representing the various data types BLAS can operate on.
pub unsafe trait Real: Num {
    fn rotm() -> unsafe extern fn(N: blasint, X: *mut Self, incX: blasint, Y: *mut Self, incY: blasint, P: *const Self);
    fn rotmg() -> unsafe extern fn( d1: *mut Self, d2: *mut Self, b1: *mut Self, b2: Self, P: *mut Self);

    fn syr() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: Self, X: *const Self, incX: blasint, A: *mut Self, lda: blasint) -> ();
    fn syr2() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: Self, X: *const Self, incX: blasint, Y: *const Self, incY: blasint, A: *mut Self, lda: blasint) -> ();
    fn symv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: Self, A: *const Self, lda: blasint, X: *const Self, incX: blasint, beta: Self, Y: *mut Self, incY: blasint) -> ();
    fn spr() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: Self, X: *const Self, incX: blasint, Ap: *mut Self) -> ();
    fn spr2() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: Self, X: *const Self, incX: blasint, Y: *const Self, incY: blasint, A: *mut Self) -> ();
    fn spmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: Self, Ap: *const Self, X: *const Self, incX: blasint, beta: Self, Y: *mut Self, incY: blasint) -> ();
    fn sbmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, K: blasint, alpha: Self, A: *const Self, lda: blasint, X: *const Self, incX: blasint, beta: Self, Y: *mut Self, incY: blasint) -> ();
}

pub unsafe trait Complex: Num {
    fn from_retself(val: <Self as Num>::RetSelf) -> Self;
    fn dotc() -> unsafe extern fn(n: blasint, x: *const <Self as Num>::Float, incx: blasint, y: *const <Self as Num>::Float, incy: blasint) -> <Self as Num>::RetSelf;
    fn dotu_sub() -> unsafe extern fn(n: blasint, x: *const <Self as Num>::Float, incx: blasint, y: *const <Self as Num>::Float, incy: blasint, ret: *mut <Self as Num>::RetSelf);
    fn dotc_sub() -> unsafe extern fn(n: blasint, x: *const <Self as Num>::Float, incx: blasint, y: *const <Self as Num>::Float, incy: blasint, ret: *mut <Self as Num>::RetSelf);

    fn her() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: <Self as Num>::Float, X: *const <Self as Num>::Float, incX: blasint, A: *mut <Self as Num>::Float, lda: blasint) -> ();
    fn her2() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: *const <Self as Num>::Float, X: *const <Self as Num>::Float, incX: blasint, Y: *const <Self as Num>::Float, incY: blasint, A: *mut <Self as Num>::Float, lda: blasint) -> ();
    fn hemv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: *const <Self as Num>::Float, A: *const <Self as Num>::Float, lda: blasint, X: *const <Self as Num>::Float, incX: blasint, beta: *const <Self as Num>::Float, Y: *mut <Self as Num>::Float, incY: blasint) -> ();
    fn hpr() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: <Self as Num>::Float, X: *const <Self as Num>::Float, incX: blasint, A: *mut <Self as Num>::Float) -> ();
    fn hpr2() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: *const <Self as Num>::Float, X: *const <Self as Num>::Float, incX: blasint, Y: *const <Self as Num>::Float, incY: blasint, Ap: *mut <Self as Num>::Float) -> ();
    fn hbmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, K: blasint, alpha: *const <Self as Num>::Float, A: *const <Self as Num>::Float, lda: blasint, X: *const <Self as Num>::Float, incX: blasint, beta: *const <Self as Num>::Float, Y: *mut <Self as Num>::Float, incY: blasint) -> ();
    fn hpmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: *const <Self as Num>::Float, Ap: *const <Self as Num>::Float, X: *const <Self as Num>::Float, incX: blasint, beta: *const <Self as Num>::Float, Y: *mut <Self as Num>::Float, incY: blasint) -> ();

    fn gemm3m() -> unsafe extern fn(Order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, TransB: CBLAS_TRANSPOSE, M: blasint, N: blasint, K: blasint, alpha: *const <Self as Num>::Float, A: *const <Self as Num>::Float, lda: blasint, B: *const <Self as Num>::Float, ldb: blasint, beta: *const <Self as Num>::Float, C: *mut <Self as Num>::Float, ldc: blasint) -> ();
    fn hemm() -> unsafe extern fn(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, M: blasint, N: blasint, alpha: *const <Self as Num>::Float, A: *const <Self as Num>::Float, lda: blasint, B: *const <Self as Num>::Float, ldb: blasint, beta: *const <Self as Num>::Float, C: *mut <Self as Num>::Float, ldc: blasint) -> ();
    fn herk() -> unsafe extern fn(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: <Self as Num>::Float, A: *const <Self as Num>::Float, lda: blasint, beta: <Self as Num>::Float, C: *mut <Self as Num>::Float, ldc: blasint) -> ();
    fn her2k() -> unsafe extern fn(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: *const <Self as Num>::Float, A: *const <Self as Num>::Float, lda: blasint, B: *const <Self as Num>::Float, ldb: blasint, beta: <Self as Num>::Float, C: *mut <Self as Num>::Float, ldc: blasint) -> ();
}

unsafe impl Num for f32 {
    type Float = f32;
    type RetSelf = f32;
    type Weird = f32;

    #[inline(always)]
    fn dot() -> unsafe extern fn(n: blasint, x: *const Self, incx: blasint, y: *const Self, incy: blasint) -> Self { cblas_sdot }
    #[inline(always)]
    fn axpy() -> unsafe extern fn(n: blasint, alpha: Self, x: *const Self, incx: blasint, y: *mut Self, incy: blasint) { cblas_saxpy }
    #[inline(always)]
    fn axpby() -> unsafe extern fn(n: blasint, alpha: Self, x: *const Self, incx: blasint, beta: Self, y: *mut Self, incy: blasint) { cblas_saxpby }
    #[inline(always)]
    fn rot() -> unsafe extern fn(N: blasint, X: *mut Self, incX: blasint, Y: *mut Self, incY: blasint, c: Self, s: Self) { cblas_srot }
    #[inline(always)]
    fn rotg() -> unsafe extern fn(a: *mut Self, b: *mut Self, c: *mut Self, s: *mut Self) { cblas_srotg }
    #[inline(always)]
    fn scal() -> unsafe extern fn(N: blasint, alpha: Self, X: *mut Self, incX: blasint) { cblas_sscal }
    #[inline(always)]
    fn asum() -> unsafe extern fn(N: blasint, x: *const Self, incx: blasint) -> Self { cblas_sasum }
    #[inline(always)]
    fn iamax() -> unsafe extern fn(n: blasint, x: *const Self, incx: blasint) -> size_t { cblas_isamax }
    #[inline(always)]
    fn nrm2() -> unsafe extern fn(N: blasint, X: *const <Self as Num>::Float, incX: blasint) -> Self { cblas_snrm2 }

    #[inline(always)]
    fn gemv() -> unsafe extern fn(order: CBLAS_ORDER, trans: CBLAS_TRANSPOSE, m: blasint, n: blasint, alpha: Self, a: *const Self, lda: blasint, x: *const Self, incx: blasint, beta: Self, y: *mut Self, incy: blasint) -> () { cblas_sgemv }
    #[inline(always)]
    fn ger() -> unsafe extern fn(order: CBLAS_ORDER, M: blasint, N: blasint, alpha: Self, X: *const Self, incX: blasint, Y: *const Self, incY: blasint, A: *mut Self, lda: blasint) -> () { cblas_sger }
    #[inline(always)]
    fn trsv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, A: *const Self, lda: blasint, X: *mut Self, incX: blasint) -> () { cblas_strsv }
    #[inline(always)]
    fn trmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, A: *const Self, lda: blasint, X: *mut Self, incX: blasint) -> () { cblas_strmv }
    #[inline(always)]
    fn tbsv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, K: blasint, A: *const Self, lda: blasint, X: *mut Self, incX: blasint) -> () { cblas_stbsv }
    #[inline(always)]
    fn tpsv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, Ap: *const Self, X: *mut Self, incX: blasint) -> () { cblas_stpsv }
    #[inline(always)]
    fn tpmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, Ap: *const Self, X: *mut Self, incX: blasint) -> () { cblas_stpmv }
    #[inline(always)]
    fn gbmv() -> unsafe extern fn(order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, M: blasint, N: blasint, KL: blasint, KU: blasint, alpha: Self, A: *const Self, lda: blasint, X: *const Self, incX: blasint, beta: Self, Y: *mut Self, incY: blasint) -> () { cblas_sgbmv }
    #[inline(always)]
    fn tbmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, K: blasint, A: *const Self, lda: blasint, X: *mut Self, incX: blasint) -> () { cblas_stbmv }

    #[inline(always)]
    fn gemm() -> unsafe extern fn(Order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, TransB: CBLAS_TRANSPOSE, M: blasint, N: blasint, K: blasint, alpha: Self, A: *const Self, lda: blasint, B: *const Self, ldb: blasint, beta: Self, C: *mut Self, ldc: blasint) -> () { cblas_sgemm }
    #[inline(always)]
    fn symm() -> unsafe extern fn(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, M: blasint, N: blasint, alpha: Self, A: *const Self, lda: blasint, B: *const Self, ldb: blasint, beta: Self, C: *mut Self, ldc: blasint) -> () { cblas_ssymm }
    #[inline(always)]
    fn syrk() -> unsafe extern fn(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: Self, A: *const Self, lda: blasint, beta: Self, C: *mut Self, ldc: blasint) -> () { cblas_ssyrk }
    #[inline(always)]
    fn syr2k() -> unsafe extern fn(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: Self, A: *const Self, lda: blasint, B: *const Self, ldb: blasint, beta: Self, C: *mut Self, ldc: blasint) -> () { cblas_ssyr2k }
    #[inline(always)]
    fn trmm() -> unsafe extern fn(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, M: blasint, N: blasint, alpha: Self, A: *const Self, lda: blasint, B: *mut Self, ldb: blasint) -> () { cblas_strmm }
    #[inline(always)]
    fn trsm() -> unsafe extern fn(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, M: blasint, N: blasint, alpha: Self, A: *const Self, lda: blasint, B: *mut Self, ldb: blasint) -> () { cblas_strsm }
}

fn c2f(a: &C) -> *const f32 {
    a as *const C as *const f32
}

unsafe extern fn caxpy_wrap(a: i32, b: num::complex::Complex<f32>, c: *const num::complex::Complex<f32>, d: i32, e: *mut num::complex::Complex<f32>, f: i32) {
    cblas_caxpy(a, c2f(&b), c as *const _, d, e as *mut f32, f);
}

unsafe extern fn caxpby_wrap(n: blasint, alpha: C, x: *const C, incx: blasint, beta: C, y: *mut C, incy: blasint) {
    cblas_caxpby(n, c2f(&alpha), x as *const f32, incx, c2f(&beta), y as *mut f32, incy)
}
unsafe extern fn crot_wrap(n: blasint, x: *mut C, incx: blasint, y: *mut C, incy: blasint, c: f32, s: C) {
    cblas_crot(n, x as *mut f32, incx, y as *mut f32, incy, c, c2f(&s));
}
unsafe extern fn cscal_wrap(n: blasint, alpha: C, x: *mut C, incx: blasint) {
    cblas_cscal(n, c2f(&alpha), x as *mut f32, incx)
}

unsafe impl Num for C {
    type Float = f32;
    type RetSelf = [f32; 2];
    type Weird = *const f32;

    #[inline(always)]
    fn dot() -> unsafe extern fn(n: blasint, x: *const <Self as Num>::Float, incx: blasint, y: *const <Self as Num>::Float, incy: blasint) -> <Self as Num>::RetSelf { cblas_cdotu }
    #[inline(always)]
    fn axpy() -> unsafe extern fn(n: blasint, alpha: Self, x: *const Self, incx: blasint, y: *mut Self, incy: blasint) { caxpy_wrap }
    #[inline(always)]
    fn axpby() -> unsafe extern fn(n: blasint, alpha: Self, x: *const Self, incx: blasint, beta: Self, y: *mut Self, incy: blasint) { caxpby_wrap }
    #[inline(always)]
    fn rot() -> unsafe extern fn(N: blasint, X: *mut Self, incX: blasint, Y: *mut Self, incY: blasint, c: <Self as Num>::Float, s: Self) { crot_wrap }
    #[inline(always)]
    fn rotg() -> unsafe extern fn(a: *mut <Self as Num>::Float, b: *mut <Self as Num>::Float, c: *mut <Self as Num>::Float, s: *mut Self) { unsafe { transmute(cblas_crotg) } }
    #[inline(always)]
    fn scal() -> unsafe extern fn(N: blasint, alpha: Self, X: *mut Self, incX: blasint) { cscal_wrap }
    #[inline(always)]
    fn asum() -> unsafe extern fn(N: blasint, x: *const <Self as Num>::Float, incx: blasint) -> <Self as Num>::Float { cblas_scasum }
    #[inline(always)]
    fn iamax() -> unsafe extern fn(n: blasint, x: *const <Self as Num>::Float, incx: blasint) -> size_t { cblas_icamax }
    #[inline(always)]
    fn nrm2() -> unsafe extern fn(N: blasint, X: *const <Self as Num>::Float, incX: blasint) -> <Self as Num>::Float { cblas_scnrm2 }

    #[inline(always)]
    fn gemv() -> unsafe extern fn(order: CBLAS_ORDER, trans: CBLAS_TRANSPOSE, m: blasint, n: blasint, alpha: <Self as Num>::Weird, a: *const <Self as Num>::Float, lda: blasint, x: *const <Self as Num>::Float, incx: blasint, beta: <Self as Num>::Weird, y: *mut <Self as Num>::Float, incy: blasint) -> () { cblas_cgemv }
    #[inline(always)]
    fn ger() -> unsafe extern fn(order: CBLAS_ORDER, M: blasint, N: blasint, alpha: <Self as Num>::Weird, X: *const <Self as Num>::Float, incX: blasint, Y: *const <Self as Num>::Float, incY: blasint, A: *mut <Self as Num>::Float, lda: blasint) -> () { cblas_cgeru }
    #[inline(always)]
    fn trsv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, A: *const <Self as Num>::Float, lda: blasint, X: *mut <Self as Num>::Float, incX: blasint) -> () { cblas_ctrsv }
    #[inline(always)]
    fn trmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, A: *const <Self as Num>::Float, lda: blasint, X: *mut <Self as Num>::Float, incX: blasint) -> () { cblas_ctrmv }
    #[inline(always)]
    fn tbsv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, K: blasint, A: *const <Self as Num>::Float, lda: blasint, X: *mut <Self as Num>::Float, incX: blasint) -> () { cblas_ctbsv }
    #[inline(always)]
    fn tpsv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, Ap: *const <Self as Num>::Float, X: *mut <Self as Num>::Float, incX: blasint) -> () { cblas_ctpsv }
    #[inline(always)]
    fn tpmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, Ap: *const <Self as Num>::Float, X: *mut <Self as Num>::Float, incX: blasint) -> () { cblas_ctpmv }
    #[inline(always)]
    fn gbmv() -> unsafe extern fn(order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, M: blasint, N: blasint, KL: blasint, KU: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, X: *const <Self as Num>::Float, incX: blasint, beta: <Self as Num>::Weird, Y: *mut <Self as Num>::Float, incY: blasint) -> () { cblas_cgbmv }
    #[inline(always)]
    fn tbmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, K: blasint, A: *const <Self as Num>::Float, lda: blasint, X: *mut <Self as Num>::Float, incX: blasint) -> () { cblas_ctbmv }

    #[inline(always)]
    fn gemm() -> unsafe extern fn(Order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, TransB: CBLAS_TRANSPOSE, M: blasint, N: blasint, K: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, B: *const <Self as Num>::Float, ldb: blasint, beta: <Self as Num>::Weird, C: *mut <Self as Num>::Float, ldc: blasint) -> () { cblas_cgemm }
    #[inline(always)]
    fn symm() -> unsafe extern fn(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, M: blasint, N: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, B: *const <Self as Num>::Float, ldb: blasint, beta: <Self as Num>::Weird, C: *mut <Self as Num>::Float, ldc: blasint) -> () { cblas_csymm }
    #[inline(always)]
    fn syrk() -> unsafe extern fn(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, beta: <Self as Num>::Weird, C: *mut <Self as Num>::Float, ldc: blasint) -> () { cblas_csyrk }
    #[inline(always)]
    fn syr2k() -> unsafe extern fn(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, B: *const <Self as Num>::Float, ldb: blasint, beta: <Self as Num>::Weird, C: *mut <Self as Num>::Float, ldc: blasint) -> () { cblas_csyr2k }
    #[inline(always)]
    fn trmm() -> unsafe extern fn(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, M: blasint, N: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, B: *mut <Self as Num>::Float, ldb: blasint) -> () { cblas_ctrmm }
    #[inline(always)]
    fn trsm() -> unsafe extern fn(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, M: blasint, N: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, B: *mut <Self as Num>::Float, ldb: blasint) -> () { cblas_ctrsm }
}

unsafe impl Complex for C {
    #[inline(always)]
    fn from_retself(val: <Self as Num>::RetSelf) -> Self {
        num::Complex {
            re: val[0],
            im: val[1],
        }
    }

    #[inline(always)]
    fn dotc() -> unsafe extern fn(n: blasint, x: *const <C as Num>::Float, incx: blasint, y: *const <C as Num>::Float, incy: blasint) -> <C as Num>::RetSelf { cblas_cdotc }
    #[inline(always)]
    fn dotu_sub() -> unsafe extern fn(n: blasint, x: *const <C as Num>::Float, incx: blasint, y: *const <C as Num>::Float, incy: blasint, ret: *mut <C as Num>::RetSelf) -> () { cblas_cdotu_sub }
    #[inline(always)]
    fn dotc_sub() -> unsafe extern fn(n: blasint, x: *const <C as Num>::Float, incx: blasint, y: *const <C as Num>::Float, incy: blasint, ret: *mut <C as Num>::RetSelf) -> () { cblas_cdotc_sub }

    #[inline(always)]
    fn her() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: <Self as Num>::Float, X: *const <Self as Num>::Float, incX: blasint, A: *mut <Self as Num>::Float, lda: blasint) -> () { cblas_cher }
    #[inline(always)]
    fn her2() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: *const <Self as Num>::Float, X: *const <Self as Num>::Float, incX: blasint, Y: *const <Self as Num>::Float, incY: blasint, A: *mut <Self as Num>::Float, lda: blasint) -> () { cblas_cher2 }
    #[inline(always)]
    fn hemv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: *const <Self as Num>::Float, A: *const <Self as Num>::Float, lda: blasint, X: *const <Self as Num>::Float, incX: blasint, beta: *const <Self as Num>::Float, Y: *mut <Self as Num>::Float, incY: blasint) -> () { cblas_chemv }
    #[inline(always)]
    fn hpr() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: <Self as Num>::Float, X: *const <Self as Num>::Float, incX: blasint, A: *mut <Self as Num>::Float) -> () { cblas_chpr }
    #[inline(always)]
    fn hpr2() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: *const <Self as Num>::Float, X: *const <Self as Num>::Float, incX: blasint, Y: *const <Self as Num>::Float, incY: blasint, Ap: *mut <Self as Num>::Float) -> () { cblas_chpr2 }
    #[inline(always)]
    fn hbmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, K: blasint, alpha: *const <Self as Num>::Float, A: *const <Self as Num>::Float, lda: blasint, X: *const <Self as Num>::Float, incX: blasint, beta: *const <Self as Num>::Float, Y: *mut <Self as Num>::Float, incY: blasint) -> () { cblas_chbmv }
    #[inline(always)]
    fn hpmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: *const <Self as Num>::Float, Ap: *const <Self as Num>::Float, X: *const <Self as Num>::Float, incX: blasint, beta: *const <Self as Num>::Float, Y: *mut <Self as Num>::Float, incY: blasint) -> () { cblas_chpmv }

    #[inline(always)]
    fn gemm3m() -> unsafe extern fn(Order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, TransB: CBLAS_TRANSPOSE, M: blasint, N: blasint, K: blasint, alpha: *const <Self as Num>::Float, A: *const <Self as Num>::Float, lda: blasint, B: *const <Self as Num>::Float, ldb: blasint, beta: *const <Self as Num>::Float, C: *mut <Self as Num>::Float, ldc: blasint) -> () { cblas_cgemm3m }
    #[inline(always)]
    fn hemm() -> unsafe extern fn(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, M: blasint, N: blasint, alpha: *const <Self as Num>::Float, A: *const <Self as Num>::Float, lda: blasint, B: *const <Self as Num>::Float, ldb: blasint, beta: *const <Self as Num>::Float, C: *mut <Self as Num>::Float, ldc: blasint) -> () { cblas_chemm }
    #[inline(always)]
    fn herk() -> unsafe extern fn(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: <Self as Num>::Float, A: *const <Self as Num>::Float, lda: blasint, beta: <Self as Num>::Float, C: *mut <Self as Num>::Float, ldc: blasint) -> () { cblas_cherk }
    #[inline(always)]
    fn her2k() -> unsafe extern fn(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: *const <Self as Num>::Float, A: *const <Self as Num>::Float, lda: blasint, B: *const <Self as Num>::Float, ldb: blasint, beta: <Self as Num>::Float, C: *mut <Self as Num>::Float, ldc: blasint) -> () { cblas_cher2k }
}

fn z2f(a: &Z) -> *const f64 {
    a as *const Z as *const f64
}

unsafe extern fn zaxpy_wrap(a: i32, b: Z, c: *const Z, d: i32, e: *mut Z, f: i32) {
    cblas_zaxpy(a, z2f(&b), c as *const f64, d, e as *mut f64, f);
}

unsafe extern fn zaxpby_wrap(n: blasint, alpha: Z, x: *const Z, incx: blasint, beta: Z, y: *mut Z, incy: blasint) {
    cblas_zaxpby(n, z2f(&alpha), x as *const f64, incx, z2f(&beta), y as *mut f64, incy)
}

unsafe extern fn zrot_wrap(n: blasint, x: *mut Z, incx: blasint, y: *mut Z, incy: blasint, c: f64, s: Z) {
    cblas_zrot(n, x as *mut f64, incx, y as *mut f64, incy, c, z2f(&s));
}

unsafe extern fn zscal_wrap(n: blasint, alpha: Z, x: *mut Z, incx: blasint) {
    cblas_zscal(n, z2f(&alpha), x as *mut f64, incx)
}

unsafe impl Num for Z {
    type Float = f64;
    type RetSelf = [f64; 2];
    type Weird = *const f64;

    #[inline(always)]
    fn dot() -> unsafe extern fn(n: blasint, x: *const <Self as Num>::Float, incx: blasint, y: *const <Self as Num>::Float, incy: blasint) -> <Self as Num>::RetSelf { cblas_zdotu }
    #[inline(always)]
    fn axpy() -> unsafe extern fn(n: blasint, alpha: Self, x: *const Self, incx: blasint, y: *mut Self, incy: blasint) { zaxpy_wrap }
    #[inline(always)]
    fn axpby() -> unsafe extern fn(n: blasint, alpha: Self, x: *const Self, incx: blasint, beta: Self, y: *mut Self, incy: blasint) { zaxpby_wrap }
    #[inline(always)]
    fn rot() -> unsafe extern fn(N: blasint, X: *mut Self, incX: blasint, Y: *mut Self, incY: blasint, c: <Self as Num>::Float, s: Self) { zrot_wrap }
    #[inline(always)]
    fn rotg() -> unsafe extern fn(a: *mut <Self as Num>::Float, b: *mut <Self as Num>::Float, c: *mut <Self as Num>::Float, s: *mut Self) { unsafe { transmute(cblas_zrotg) } }
    #[inline(always)]
    fn scal() -> unsafe extern fn(N: blasint, alpha: Self, X: *mut Self, incX: blasint) { zscal_wrap }
    #[inline(always)]
    fn asum() -> unsafe extern fn(N: blasint, x: *const <Self as Num>::Float, incx: blasint) -> <Self as Num>::Float { cblas_dzasum }
    #[inline(always)]
    fn iamax() -> unsafe extern fn(n: blasint, x: *const <Self as Num>::Float, incx: blasint) -> size_t { cblas_izamax }
    #[inline(always)]
    fn nrm2() -> unsafe extern fn(N: blasint, X: *const <Self as Num>::Float, incX: blasint) -> <Self as Num>::Float { cblas_dznrm2 }

    #[inline(always)]
    fn gemv() -> unsafe extern fn(order: CBLAS_ORDER, trans: CBLAS_TRANSPOSE, m: blasint, n: blasint, alpha: <Self as Num>::Weird, a: *const <Self as Num>::Float, lda: blasint, x: *const <Self as Num>::Float, incx: blasint, beta: <Self as Num>::Weird, y: *mut <Self as Num>::Float, incy: blasint) -> () { cblas_zgemv }
    #[inline(always)]
    fn ger() -> unsafe extern fn(order: CBLAS_ORDER, M: blasint, N: blasint, alpha: <Self as Num>::Weird, X: *const <Self as Num>::Float, incX: blasint, Y: *const <Self as Num>::Float, incY: blasint, A: *mut <Self as Num>::Float, lda: blasint) -> () { cblas_zgeru }
    #[inline(always)]
    fn trsv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, A: *const <Self as Num>::Float, lda: blasint, X: *mut <Self as Num>::Float, incX: blasint) -> () { cblas_ztrsv }
    #[inline(always)]
    fn trmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, A: *const <Self as Num>::Float, lda: blasint, X: *mut <Self as Num>::Float, incX: blasint) -> () { cblas_ztrmv }
    #[inline(always)]
    fn tbsv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, K: blasint, A: *const <Self as Num>::Float, lda: blasint, X: *mut <Self as Num>::Float, incX: blasint) -> () { cblas_ztbsv }
    #[inline(always)]
    fn tpsv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, Ap: *const <Self as Num>::Float, X: *mut <Self as Num>::Float, incX: blasint) -> () { cblas_ztpsv }
    #[inline(always)]
    fn tpmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, Ap: *const <Self as Num>::Float, X: *mut <Self as Num>::Float, incX: blasint) -> () { cblas_ztpmv }
    #[inline(always)]
    fn gbmv() -> unsafe extern fn(order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, M: blasint, N: blasint, KL: blasint, KU: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, X: *const <Self as Num>::Float, incX: blasint, beta: <Self as Num>::Weird, Y: *mut <Self as Num>::Float, incY: blasint) -> () { cblas_zgbmv }
    #[inline(always)]
    fn tbmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, K: blasint, A: *const <Self as Num>::Float, lda: blasint, X: *mut <Self as Num>::Float, incX: blasint) -> () { cblas_ztbmv }

    #[inline(always)]
    fn gemm() -> unsafe extern fn(Order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, TransB: CBLAS_TRANSPOSE, M: blasint, N: blasint, K: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, B: *const <Self as Num>::Float, ldb: blasint, beta: <Self as Num>::Weird, C: *mut <Self as Num>::Float, ldc: blasint) -> () { cblas_zgemm }
    #[inline(always)]
    fn symm() -> unsafe extern fn(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, M: blasint, N: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, B: *const <Self as Num>::Float, ldb: blasint, beta: <Self as Num>::Weird, C: *mut <Self as Num>::Float, ldc: blasint) -> () { cblas_zsymm }
    #[inline(always)]
    fn syrk() -> unsafe extern fn(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, beta: <Self as Num>::Weird, C: *mut <Self as Num>::Float, ldc: blasint) -> () { cblas_zsyrk }
    #[inline(always)]
    fn syr2k() -> unsafe extern fn(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, B: *const <Self as Num>::Float, ldb: blasint, beta: <Self as Num>::Weird, C: *mut <Self as Num>::Float, ldc: blasint) -> () { cblas_zsyr2k }
    #[inline(always)]
    fn trmm() -> unsafe extern fn(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, M: blasint, N: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, B: *mut <Self as Num>::Float, ldb: blasint) -> () { cblas_ztrmm }
    #[inline(always)]
    fn trsm() -> unsafe extern fn(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, M: blasint, N: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, B: *mut <Self as Num>::Float, ldb: blasint) -> () { cblas_ztrsm }
}

unsafe impl Complex for Z {
    #[inline(always)]
    fn from_retself(val: <Self as Num>::RetSelf) -> Self {
        num::Complex {
            re: val[0],
            im: val[1],
        }
    }
    #[inline(always)]
    fn dotc() -> unsafe extern fn(n: blasint, x: *const <Z as Num>::Float, incx: blasint, y: *const <Z as Num>::Float, incy: blasint) -> <Z as Num>::RetSelf { cblas_zdotc }
    #[inline(always)]
    fn dotu_sub() -> unsafe extern fn(n: blasint, x: *const <Z as Num>::Float, incx: blasint, y: *const <Z as Num>::Float, incy: blasint, ret: *mut <Z as Num>::RetSelf) -> () { cblas_zdotu_sub }
    #[inline(always)]
    fn dotc_sub() -> unsafe extern fn(n: blasint, x: *const <Z as Num>::Float, incx: blasint, y: *const <Z as Num>::Float, incy: blasint, ret: *mut <Z as Num>::RetSelf) -> () { cblas_zdotc_sub }

    #[inline(always)]
    fn her() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: <Self as Num>::Float, X: *const <Self as Num>::Float, incX: blasint, A: *mut <Self as Num>::Float, lda: blasint) -> () { cblas_zher }
    #[inline(always)]
    fn her2() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: *const <Self as Num>::Float, X: *const <Self as Num>::Float, incX: blasint, Y: *const <Self as Num>::Float, incY: blasint, A: *mut <Self as Num>::Float, lda: blasint) -> () { cblas_zher2 }
    #[inline(always)]
    fn hemv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: *const <Self as Num>::Float, A: *const <Self as Num>::Float, lda: blasint, X: *const <Self as Num>::Float, incX: blasint, beta: *const <Self as Num>::Float, Y: *mut <Self as Num>::Float, incY: blasint) -> () { cblas_zhemv }
    #[inline(always)]
    fn hpr() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: <Self as Num>::Float, X: *const <Self as Num>::Float, incX: blasint, A: *mut <Self as Num>::Float) -> () { cblas_zhpr }
    #[inline(always)]
    fn hpr2() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: *const <Self as Num>::Float, X: *const <Self as Num>::Float, incX: blasint, Y: *const <Self as Num>::Float, incY: blasint, Ap: *mut <Self as Num>::Float) -> () { cblas_zhpr2 }
    #[inline(always)]
    fn hbmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, K: blasint, alpha: *const <Self as Num>::Float, A: *const <Self as Num>::Float, lda: blasint, X: *const <Self as Num>::Float, incX: blasint, beta: *const <Self as Num>::Float, Y: *mut <Self as Num>::Float, incY: blasint) -> () { cblas_zhbmv }
    #[inline(always)]
    fn hpmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: *const <Self as Num>::Float, Ap: *const <Self as Num>::Float, X: *const <Self as Num>::Float, incX: blasint, beta: *const <Self as Num>::Float, Y: *mut <Self as Num>::Float, incY: blasint) -> () { cblas_zhpmv }

    #[inline(always)]
    fn gemm3m() -> unsafe extern fn(Order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, TransB: CBLAS_TRANSPOSE, M: blasint, N: blasint, K: blasint, alpha: *const <Self as Num>::Float, A: *const <Self as Num>::Float, lda: blasint, B: *const <Self as Num>::Float, ldb: blasint, beta: *const <Self as Num>::Float, C: *mut <Self as Num>::Float, ldc: blasint) -> () { cblas_zgemm3m }
    #[inline(always)]
    fn hemm() -> unsafe extern fn(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, M: blasint, N: blasint, alpha: *const <Self as Num>::Float, A: *const <Self as Num>::Float, lda: blasint, B: *const <Self as Num>::Float, ldb: blasint, beta: *const <Self as Num>::Float, C: *mut <Self as Num>::Float, ldc: blasint) -> () { cblas_zhemm }
    #[inline(always)]
    fn herk() -> unsafe extern fn(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: <Self as Num>::Float, A: *const <Self as Num>::Float, lda: blasint, beta: <Self as Num>::Float, C: *mut <Self as Num>::Float, ldc: blasint) -> () { cblas_zherk }
    #[inline(always)]
    fn her2k() -> unsafe extern fn(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: *const <Self as Num>::Float, A: *const <Self as Num>::Float, lda: blasint, B: *const <Self as Num>::Float, ldb: blasint, beta: <Self as Num>::Float, C: *mut <Self as Num>::Float, ldc: blasint) -> () { cblas_zher2k }
}

unsafe impl Real for f32 {
    #[inline(always)]
    fn rotm() -> unsafe extern fn(N: blasint, X: *mut Self, incX: blasint, Y: *mut Self, incY: blasint, P: *const Self) { cblas_srotm }
    #[inline(always)]
    fn rotmg() -> unsafe extern fn( d1: *mut Self, d2: *mut Self, b1: *mut Self, b2: Self, P: *mut Self) { cblas_srotmg }

    #[inline(always)]
    fn syr() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: Self, X: *const Self, incX: blasint, A: *mut Self, lda: blasint) -> () { cblas_ssyr }
    #[inline(always)]
    fn syr2() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: Self, X: *const Self, incX: blasint, Y: *const Self, incY: blasint, A: *mut Self, lda: blasint) -> () { cblas_ssyr2 }
    #[inline(always)]
    fn symv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: Self, A: *const Self, lda: blasint, X: *const Self, incX: blasint, beta: Self, Y: *mut Self, incY: blasint) -> () { cblas_ssymv }
    #[inline(always)]
    fn spr() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: Self, X: *const Self, incX: blasint, Ap: *mut Self) -> () { cblas_sspr }
    #[inline(always)]
    fn spr2() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: Self, X: *const Self, incX: blasint, Y: *const Self, incY: blasint, A: *mut Self) -> () { cblas_sspr2 }
    #[inline(always)]
    fn spmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: Self, Ap: *const Self, X: *const Self, incX: blasint, beta: Self, Y: *mut Self, incY: blasint) -> () { cblas_sspmv }
    #[inline(always)]
    fn sbmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, K: blasint, alpha: Self, A: *const Self, lda: blasint, X: *const Self, incX: blasint, beta: Self, Y: *mut Self, incY: blasint) -> () { cblas_ssbmv }
}

unsafe impl Num for f64 {
    type Float = f64;
    type RetSelf = f64;
    type Weird = f64;

    #[inline(always)]
    fn dot() -> unsafe extern fn(n: blasint, x: *const Self, incx: blasint, y: *const Self, incy: blasint) -> Self { cblas_ddot }
    #[inline(always)]
    fn axpy() -> unsafe extern fn(n: blasint, alpha: Self, x: *const Self, incx: blasint, y: *mut Self, incy: blasint) { cblas_daxpy }
    #[inline(always)]
    fn axpby() -> unsafe extern fn(n: blasint, alpha: Self, x: *const Self, incx: blasint, beta: Self, y: *mut Self, incy: blasint) { cblas_daxpby }
    #[inline(always)]
    fn rot() -> unsafe extern fn(N: blasint, X: *mut Self, incX: blasint, Y: *mut Self, incY: blasint, c: Self, s: Self) { cblas_drot }
    #[inline(always)]
    fn rotg() -> unsafe extern fn(a: *mut Self, b: *mut Self, c: *mut Self, s: *mut Self) { cblas_drotg }
    #[inline(always)]
    fn scal() -> unsafe extern fn(N: blasint, alpha: Self, X: *mut Self, incX: blasint) { cblas_dscal }
    #[inline(always)]
    fn asum() -> unsafe extern fn(N: blasint, x: *const Self, incx: blasint) -> Self { cblas_dasum }
    #[inline(always)]
    fn iamax() -> unsafe extern fn(n: blasint, x: *const Self, incx: blasint) -> size_t { cblas_idamax }
    #[inline(always)]
    fn nrm2() -> unsafe extern fn(N: blasint, X: *const Self, incX: blasint) -> Self { cblas_dnrm2 }

    #[inline(always)]
    fn gemv() -> unsafe extern fn(order: CBLAS_ORDER, trans: CBLAS_TRANSPOSE, m: blasint, n: blasint, alpha: Self, a: *const Self, lda: blasint, x: *const Self, incx: blasint, beta: Self, y: *mut Self, incy: blasint) -> () { cblas_dgemv }
    #[inline(always)]
    fn ger() -> unsafe extern fn(order: CBLAS_ORDER, M: blasint, N: blasint, alpha: Self, X: *const Self, incX: blasint, Y: *const Self, incY: blasint, A: *mut Self, lda: blasint) -> () { cblas_dger }
    #[inline(always)]
    fn trsv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, A: *const Self, lda: blasint, X: *mut Self, incX: blasint) -> () { cblas_dtrsv }
    #[inline(always)]
    fn trmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, A: *const Self, lda: blasint, X: *mut Self, incX: blasint) -> () { cblas_dtrmv }
    #[inline(always)]
    fn tbsv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, K: blasint, A: *const Self, lda: blasint, X: *mut Self, incX: blasint) -> () { cblas_dtbsv }
    #[inline(always)]
    fn tpsv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, Ap: *const Self, X: *mut Self, incX: blasint) -> () { cblas_dtpsv }
    #[inline(always)]
    fn tpmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, Ap: *const Self, X: *mut Self, incX: blasint) -> () { cblas_dtpmv }
    #[inline(always)]
    fn gbmv() -> unsafe extern fn(order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, M: blasint, N: blasint, KL: blasint, KU: blasint, alpha: Self, A: *const Self, lda: blasint, X: *const Self, incX: blasint, beta: Self, Y: *mut Self, incY: blasint) -> () { cblas_dgbmv }
    #[inline(always)]
    fn tbmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, K: blasint, A: *const Self, lda: blasint, X: *mut Self, incX: blasint) -> () { cblas_dtbmv }


    #[inline(always)]
    fn gemm() -> unsafe extern fn(Order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, TransB: CBLAS_TRANSPOSE, M: blasint, N: blasint, K: blasint, alpha: Self, A: *const Self, lda: blasint, B: *const Self, ldb: blasint, beta: Self, C: *mut Self, ldc: blasint) -> () { cblas_dgemm }
    #[inline(always)]
    fn symm() -> unsafe extern fn(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, M: blasint, N: blasint, alpha: Self, A: *const Self, lda: blasint, B: *const Self, ldb: blasint, beta: Self, C: *mut Self, ldc: blasint) -> () { cblas_dsymm }
    #[inline(always)]
    fn syrk() -> unsafe extern fn(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: Self, A: *const Self, lda: blasint, beta: Self, C: *mut Self, ldc: blasint) -> () { cblas_dsyrk }
    #[inline(always)]
    fn syr2k() -> unsafe extern fn(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: Self, A: *const Self, lda: blasint, B: *const Self, ldb: blasint, beta: Self, C: *mut Self, ldc: blasint) -> () { cblas_dsyr2k }
    #[inline(always)]
    fn trmm() -> unsafe extern fn(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, M: blasint, N: blasint, alpha: Self, A: *const Self, lda: blasint, B: *mut Self, ldb: blasint) -> () { cblas_dtrmm }
    #[inline(always)]
    fn trsm() -> unsafe extern fn(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, M: blasint, N: blasint, alpha: Self, A: *const Self, lda: blasint, B: *mut Self, ldb: blasint) -> () { cblas_dtrsm }
}

unsafe impl Real for f64 {
    #[inline(always)]
    fn rotm() -> unsafe extern fn(N: blasint, X: *mut Self, incX: blasint, Y: *mut Self, incY: blasint, P: *const Self) { cblas_drotm }
    #[inline(always)]
    fn rotmg() -> unsafe extern fn( d1: *mut Self, d2: *mut Self, b1: *mut Self, b2: Self, P: *mut Self) { cblas_drotmg }

    #[inline(always)]
    fn syr() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: Self, X: *const Self, incX: blasint, A: *mut Self, lda: blasint) -> () { cblas_dsyr }
    #[inline(always)]
    fn syr2() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: Self, X: *const Self, incX: blasint, Y: *const Self, incY: blasint, A: *mut Self, lda: blasint) -> () { cblas_dsyr2 }
    #[inline(always)]
    fn symv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: Self, A: *const Self, lda: blasint, X: *const Self, incX: blasint, beta: Self, Y: *mut Self, incY: blasint) -> () { cblas_dsymv }
    #[inline(always)]
    fn spr() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: Self, X: *const Self, incX: blasint, Ap: *mut Self) -> () { cblas_dspr }
    #[inline(always)]
    fn spr2() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: Self, X: *const Self, incX: blasint, Y: *const Self, incY: blasint, A: *mut Self) -> () { cblas_dspr2 }
    #[inline(always)]
    fn spmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: Self, Ap: *const Self, X: *const Self, incX: blasint, beta: Self, Y: *mut Self, incY: blasint) -> () { cblas_dspmv }
    #[inline(always)]
    fn sbmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, K: blasint, alpha: Self, A: *const Self, lda: blasint, X: *const Self, incX: blasint, beta: Self, Y: *mut Self, incY: blasint) -> () { cblas_dsbmv }
}

pub trait Vector {
    type Element: Num;

    fn len(&self) -> blasint;
    fn stride(&self) -> blasint;
    fn as_ptr(&self) -> *const Self::Element;
    fn as_mut_ptr(&mut self) -> *mut Self::Element;
}

impl<T: Num> Vector for [T] {
    type Element = T;

    #[inline(always)]
    fn len(&self) -> blasint {
        SliceExt::len(self) as blasint
    }

    #[inline(always)]
    fn stride(&self) -> blasint {
        1
    }

    #[inline(always)]
    fn as_ptr(&self) -> *const <[T] as Vector>::Element {
        SliceExt::as_ptr(self)
    }

    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut <[T] as Vector>::Element {
        SliceExt::as_mut_ptr(self)
    }
}


/// x · y
#[inline(always)]
pub fn dot<V>(x: V, y: V) -> <V::Element as Num>::RetSelf where V: Vector {
    debug_assert!(x.len() == y.len());
    let len = min(x.len(), y.len());
    unsafe { V::Element::dot()(len, x.as_ptr() as *const _, x.stride(), y.as_ptr() as *const _, y.stride()) }
}

/// y += a * x
#[inline(always)]
pub fn axpy<V, U>(alpha: V::Element, x: &mut V, y: &mut U) where V: Vector, U: Vector<Element = V::Element>  {
    debug_assert!(x.len() == y.len());
    let len = min(x.len(), y.len());
    unsafe { V::Element::axpy()(len, alpha, x.as_ptr(), x.stride(), y.as_mut_ptr(), y.stride()) }
}

/// Linear combination: y = a * x + b * y
#[inline(always)]
pub fn axpby<V, U>(alpha: V::Element, x: &mut V, beta: U::Element, y: &mut U) where V: Vector, U: Vector<Element = V::Element>  {
    debug_assert!(x.len() == y.len());
    let len = min(x.len(), y.len());
    unsafe { V::Element::axpby()(len, alpha, x.as_ptr(), x.stride(), beta, y.as_mut_ptr(), y.stride()) }
}

/// Do something to x and y involving a Givens rotation where s = sinθ and c = cosθ?

#[inline(always)]
pub fn rot<V>(x: &mut V, y: &mut V, s: <V::Element as Num>::Float, c: V::Element) where V: Vector{
    debug_assert!(x.len() == y.len());
    let len = min(x.len(), y.len());
    unsafe { V::Element::rot()(len, x.as_mut_ptr(), x.stride(), y.as_mut_ptr(), y.stride(), s, c) }
}

/// Setup a Givens rotation in the passed vector <a, b>, and returning new elements (c, s).
#[inline(always)]
pub fn rotg<V>(x: &mut V) -> (<V::Element as Num>::Float, V::Element) where V: Vector {
    // a, b, s COMPLEX, c REAL
    assert!(x.len() >= 2);
    debug_assert!(x.len() == 2);

    let mut s: V::Element = unsafe { std::mem::zeroed() };
    let mut c: <V::Element as Num>::Float = unsafe { std::mem::zeroed() };
    unsafe { V::Element::rotg()(x.as_mut_ptr() as *mut _, x.as_mut_ptr().offset(1) as *mut _, &mut c, &mut s) }
    (c, s)
}

/// array *= alpha
///
/// *Note:* This crate does not expose zdscal or csscal because they are not actually implemented
/// specially in OpenBLAS.
#[inline(always)]
pub fn scal<V>(alpha: V::Element, x: &mut V) where V: Vector {
    unsafe { V::Element::scal()(x.len(), alpha, x.as_mut_ptr(), x.stride()) }
}

/// Sum of the absolute values of the vector's elements.
///
/// For a complex vector, the "absolute value" is `abs(real) + abs(imag)`
#[inline(always)]
pub fn asum<V>(x: &V) -> <V::Element as Num>::Float where V: Vector {
    unsafe { V::Element::asum()(x.len(), x.as_ptr() as *const _, x.stride()) }
}

/// Index of the first value in the vector with the largest absolute value.
#[inline(always)]
pub fn iamax<V>(x: &V) -> size_t where V: Vector {
    unsafe { V::Element::iamax()(x.len(), x.as_ptr() as *const _, x.stride()) }
}

/// L2 norm of the vector `(sqrt(sum(|x_i|^2)))`, where |x_i| is the complex modulus for a complex number, absolute value otherwise.
#[inline(always)]
pub fn nrm2<V>(x: &V) -> <V::Element as Num>::Float where V: Vector {
    unsafe { V::Element::nrm2()(x.len(), x.as_ptr() as *const _, x.stride()) }
}

/// Do something *really* strange involving a "modified Givens rotation"
pub fn rotm<V, U>(x: &mut V, y: &mut U, param: &[V::Element; 5]) where V: Vector, U: Vector<Element = V::Element>, V::Element: Real {
    debug_assert!(x.len() == y.len());
    let len = min(x.len(), y.len());
    unsafe { V::Element::rotm()(len, x.as_mut_ptr(), x.stride(), y.as_mut_ptr(), y.stride(), param as *const _ as *const _) }
}

/// Setup a "modified Givens rotation", where the parameters and return value are beyond my understanding.
///
/// I think this won't modify the `coord` vector, but I can't really tell from the CBLAS interface
/// whether it does things beyond the Fortran definition.
#[inline(always)]
pub fn rotmg<V, U>(diag: &mut V, coord: &mut U) -> [V::Element; 5] where V: Vector, U: Vector<Element = V::Element>, V::Element: Real {
    // a, b, s COMPLEX, c REAL
    assert!(diag.len() >= 2);
    assert!(coord.len() >= 2);
    debug_assert!(diag.len() == 2);
    debug_assert!(coord.len() == 2);

    let mut param: [V::Element; 5] = unsafe { std::mem::zeroed() };

    unsafe { V::Element::rotmg()(diag.as_mut_ptr() as *mut _, diag.as_mut_ptr().offset(1) as *mut _,
                                 coord.as_mut_ptr() as *mut _, *(coord.as_ptr().offset(1) as *const _),
                                 &mut param as *mut _ as *mut _) }

    param
}

/// Hermitian inner product of the complex vectors
#[inline(always)]
pub fn dotc<V, U>(x: &V, y: &U) -> V::Element where V: Vector, U: Vector<Element = V::Element>, V::Element: Complex {
    debug_assert!(x.len() == y.len());
    let len = min(x.len(), y.len());

    V::Element::from_retself(unsafe { V::Element::dotc()(len, x.as_ptr() as *const _, x.stride(), y.as_ptr() as *const _, y.stride()) })
}


#[cfg(test)]
mod benches {
    use std::iter::repeat;

    #[bench]
    fn dgemv_few_large(bench: &mut ::test::Bencher) {
        let m = 1000;

        let a = repeat(1.0).take(m * m).collect::<Vec<_>>();
        let x = repeat(1.0).take(m).collect::<Vec<_>>();
        let mut y = repeat(1.0).take(m).collect::<Vec<_>>();

        bench.iter(|| {
            ::dgemv(::Layout::ColumnMajor, ::Transpose::None,
                    m, m, 1.0, &a, m, &x, 1, 1.0, &mut y, 1)
        });
    }

    #[bench]
    fn dgemv_many_small(bench: &mut ::test::Bencher) {
        let m = 20;

        let a = repeat(1.0).take(m * m).collect::<Vec<_>>();
        let x = repeat(1.0).take(m).collect::<Vec<_>>();
        let mut y = repeat(1.0).take(m).collect::<Vec<_>>();

        bench.iter(|| {
            for _ in 0..20000 {
                ::dgemv(::Layout::ColumnMajor, ::Transpose::None,
                        m, m, 1.0, &a, m, &x, 1, 1.0, &mut y, 1);
            }
        });
    }
}
