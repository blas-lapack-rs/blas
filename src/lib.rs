//! An interface to the [Basic Linear Algebra Subprograms][1].
//!
//! Note: level 3 (matrix-matrix) functions are marked unsafe, as the matrix traits are not
//! finalized, and I am suspect of their correctness.
//!
//! [1]: http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms

#![allow(unused_unsafe)]

extern crate num;
extern crate libc;

extern crate "libblas-sys" as raw;

type C = num::Complex<f32>;
type Z = num::Complex<f64>;

use raw::*;
use libc::size_t;
use std::mem::transmute;
use std::cmp::min;
use std::ops::{Deref, DerefMut, Add, Sub, Mul, Div};

#[repr(C)]
#[derive(Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum Layout {
    RowMajor = CblasRowMajor as isize,
    ColumnMajor = CblasColMajor as isize,
}

#[repr(C)]
#[derive(Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum Transpose {
    NoTrans = CblasNoTrans as isize,
    Trans = CblasTrans as isize,
    ConjTrans = CblasConjTrans as isize,
}

#[repr(C)]
#[derive(Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum Uplo {
    Upper = CblasUpper as isize,
    Lower = CblasLower as isize,
}

#[repr(C)]
#[derive(Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum Diag {
    NonUnit = CblasNonUnit as isize,
    Unit = CblasUnit as isize,
}

#[repr(C)]
#[derive(Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum Side {
    Left = CblasLeft as isize,
    Right = CblasRight as isize,
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
pub unsafe trait Num: Copy + Add<Output=Self> + Sub<Output=Self> + Div<Output=Self> + Mul<Output=Self> + PartialEq + num::Zero {
    type Float: Copy;
    type RetSelf: Copy;
    type Weird: Copy;

    fn as_weird(&self) -> Self::Weird;
    fn from_retself(Self::RetSelf) -> Self;

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
pub unsafe trait Real: Num<Float = Self, RetSelf = Self, Weird = Self> + std::num::Float {
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
    fn dotc() -> unsafe extern fn(n: blasint, x: *const <Self as Num>::Float, incx: blasint, y: *const <Self as Num>::Float, incy: blasint) -> <Self as Num>::RetSelf;
    fn dotu_sub() -> unsafe extern fn(n: blasint, x: *const <Self as Num>::Float, incx: blasint, y: *const <Self as Num>::Float, incy: blasint, ret: *mut <Self as Num>::RetSelf);
    fn dotc_sub() -> unsafe extern fn(n: blasint, x: *const <Self as Num>::Float, incx: blasint, y: *const <Self as Num>::Float, incy: blasint, ret: *mut <Self as Num>::RetSelf);

    fn her() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: <Self as Num>::Float, X: *const <Self as Num>::Float, incX: blasint, A: *mut <Self as Num>::Float, lda: blasint) -> ();
    fn her2() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: <Self as Num>::Weird, X: *const <Self as Num>::Float, incX: blasint, Y: *const <Self as Num>::Float, incY: blasint, A: *mut <Self as Num>::Float, lda: blasint) -> ();
    fn hemv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, X: *const <Self as Num>::Float, incX: blasint, beta: <Self as Num>::Weird, Y: *mut <Self as Num>::Float, incY: blasint) -> ();
    fn hpr() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: <Self as Num>::Float, X: *const <Self as Num>::Float, incX: blasint, A: *mut <Self as Num>::Float) -> ();
    fn hpr2() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: <Self as Num>::Weird, X: *const <Self as Num>::Float, incX: blasint, Y: *const <Self as Num>::Float, incY: blasint, Ap: *mut <Self as Num>::Float) -> ();
    fn hbmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, K: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, X: *const <Self as Num>::Float, incX: blasint, beta: <Self as Num>::Weird, Y: *mut <Self as Num>::Float, incY: blasint) -> ();
    fn hpmv() -> unsafe extern fn(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: <Self as Num>::Weird, Ap: *const <Self as Num>::Float, X: *const <Self as Num>::Float, incX: blasint, beta: <Self as Num>::Weird, Y: *mut <Self as Num>::Float, incY: blasint) -> ();

    fn gemm3m() -> unsafe extern fn(Order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, TransB: CBLAS_TRANSPOSE, M: blasint, N: blasint, K: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, B: *const <Self as Num>::Float, ldb: blasint, beta: <Self as Num>::Weird, C: *mut <Self as Num>::Float, ldc: blasint) -> ();
    fn hemm() -> unsafe extern fn(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, M: blasint, N: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, B: *const <Self as Num>::Float, ldb: blasint, beta: <Self as Num>::Weird, C: *mut <Self as Num>::Float, ldc: blasint) -> ();
    fn herk() -> unsafe extern fn(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: <Self as Num>::Float, A: *const <Self as Num>::Float, lda: blasint, beta: <Self as Num>::Float, C: *mut <Self as Num>::Float, ldc: blasint) -> ();
    fn her2k() -> unsafe extern fn(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: <Self as Num>::Weird, A: *const <Self as Num>::Float, lda: blasint, B: *const <Self as Num>::Float, ldb: blasint, beta: <Self as Num>::Float, C: *mut <Self as Num>::Float, ldc: blasint) -> ();
}

unsafe impl Num for f32 {
    type Float = f32;
    type RetSelf = f32;
    type Weird = f32;

    #[inline(always)]
    fn as_weird(&self) -> f32 { *self }
    #[inline(always)]
    fn from_retself(x: f32) -> f32 { x }

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
    fn as_weird(&self) -> *const f32 { self as *const _ as *const _ }
    #[inline(always)]
    fn from_retself(x: [f32; 2]) -> C { num::Complex { re: x[0], im: x[1] } }

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
    fn as_weird(&self) -> *const f64 { self as *const _ as *const _ }
    #[inline(always)]
    fn from_retself(x: [f64; 2]) -> Z { num::Complex { re: x[0], im: x[1] } }

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
    fn as_weird(&self) -> f64 { *self }
    #[inline(always)]
    fn from_retself(x: f64) -> f64 { x }

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

pub unsafe trait Vector {
    type Element: Num;

    /// Number of elements in the vector.
    fn len(&self) -> blasint;

    /// The number of elements between consecutive vector entries.
    ///
    /// This is *not* in bytes!
    fn stride(&self) -> blasint;

    fn as_ptr(&self) -> *const Self::Element;
    fn as_mut_ptr(&mut self) -> *mut Self::Element;
}

unsafe impl<T: Num> Vector for [T] {
    type Element = T;

    #[inline(always)]
    fn len(&self) -> blasint {
        <[T]>::len(self) as blasint
    }

    #[inline(always)]
    fn stride(&self) -> blasint {
        1
    }

    #[inline(always)]
    fn as_ptr(&self) -> *const <[T] as Vector>::Element {
        <[T]>::as_ptr(self)
    }

    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut <[T] as Vector>::Element {
        <[T]>::as_mut_ptr(self)
    }
}

pub unsafe trait Matrix {
    type Element: Num;

    /// (m, n) where the matrix has M rows and N columns.
    fn dim(&self) -> (blasint, blasint);

    /// The "stride" of the major ("leading") dimension.
    ///
    /// If the matrix is row-major, then this is the number of elements between entries in adjacent
    /// rows with the same column index. This can be useful when "slicing" a portion of a matrix,
    /// but is usually just going to be the number of rows/columns.
    fn major_stride(&self) -> blasint {
        let (m, n) = self.dim();
        match self.layout() {
            Layout::RowMajor => n,
            Layout::ColumnMajor => m,
        }
    }

    fn layout(&self) -> Layout { Layout::RowMajor }
    fn transpose(&self) -> Transpose { Transpose::NoTrans }
    fn uplo(&self) -> Uplo { Uplo::Upper }
    fn diag(&self) -> Diag { Diag::NonUnit }

    fn as_ptr(&self) -> *const Self::Element;
    fn as_mut_ptr(&mut self) -> *mut Self::Element;
}

/// A band matrix is a special form of sparse matrix.
///
/// A band matrix is zero everywhere except maybe along the diagonal, on elements up to `kl` rows below
/// the diagonal, and elements up to `ku` rows above the diagonal. Note that when kl = ku = 0, only
/// the diagonal is stored, which can be useful.
pub trait BandMatrix: Matrix {
    /// Number of sub-diagonals.
    fn kl(&self) -> blasint;
    /// Number of super-diagonals.
    fn ku(&self) -> blasint;
}

/// A packed matrix is a special form of sparse matrix.
///
/// A packed matrix is zero everywhere except maybe the triangular portion indicated by
/// `Matrix::uplo`, which is stored column-by-column.
pub trait PackedMatrix: Matrix {

}

/// A Matrix whose data is stored in a Vec.
///
/// The size of the matrix is frozen for as long as this struct exists; getting the number of
/// rows/cols wrong causes BLAS to read/write out-of-bounds.
pub struct VecMatrix<T> {
    rows: blasint,
    cols: blasint,
    data: Vec<T>,
    pub tran: Transpose,
    pub uplo: Uplo,
    pub diag: Diag,
}

impl<T> VecMatrix<T> {
    /// Create a non-transposed, upper, non-unit matrix.
    pub fn from_parts(rows: blasint, cols: blasint, data: Vec<T>) -> VecMatrix<T> {
        VecMatrix {
            rows: rows,
            cols: cols,
            data: data,
            tran: Transpose::NoTrans,
            uplo: Uplo::Upper,
            diag: Diag::NonUnit,
        }
    }

    pub fn unwrap(self) -> Vec<T> {
        self.data
    }
}

unsafe impl<T: Num> Matrix for VecMatrix<T> {
    type Element = T;

    fn dim(&self) -> (blasint, blasint) {
        if self.tran == Transpose::NoTrans {
            (self.rows, self.cols)
        } else {
            (self.cols, self.rows)
        }
    }

    fn transpose(&self) -> Transpose { self.tran }
    fn uplo(&self) -> Uplo { self.uplo }
    fn diag(&self) -> Diag { self.diag }

    fn as_ptr(&self) -> *const T {
        <[T]>::as_ptr(&self.data[..])
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        <[T]>::as_mut_ptr(&mut self.data[..])
    }
}

impl<T> Deref for VecMatrix<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.data
    }
}

impl<T> DerefMut for VecMatrix<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.data[..]
    }
}

/// x · y
#[inline(always)]
pub fn dot<V: ?Sized>(x: &V, y: &V) -> V::Element where V: Vector {
    debug_assert_eq!(x.len(), y.len());
    let len = min(x.len(), y.len());
    V::Element::from_retself(unsafe { V::Element::dot()(len, x.as_ptr() as *const _, x.stride(), y.as_ptr() as *const _, y.stride()) })
}

/// y += a * x
#[inline(always)]
pub fn axpy<V: ?Sized, U: ?Sized>(alpha: V::Element, x: &V, y: &mut U) where V: Vector, U: Vector<Element = V::Element>  {
    debug_assert_eq!(x.len(), y.len());
    let len = min(x.len(), y.len());
    unsafe { V::Element::axpy()(len, alpha, x.as_ptr(), x.stride(), y.as_mut_ptr(), y.stride()) }
}

/// Linear combination: y = a * x + b * y
#[inline(always)]
pub fn axpby<V: ?Sized, U: ?Sized>(alpha: V::Element, x: &V, beta: U::Element, y: &mut U) where V: Vector, U: Vector<Element = V::Element>  {
    debug_assert_eq!(x.len(), y.len());
    let len = min(x.len(), y.len());
    unsafe { V::Element::axpby()(len, alpha, x.as_ptr(), x.stride(), beta, y.as_mut_ptr(), y.stride()) }
}

/// Rotate each (x, y) pair pulling coordinates from `x` and `y`, overwriting them, by doing a
/// matrix multiplication: `[x, y] = [[c, s], [-s, c]] * [x, y]`
#[inline(always)]
pub fn rot<V: ?Sized>(x: &mut V, y: &mut V, c: <V::Element as Num>::Float, s: V::Element) where V: Vector{
    debug_assert_eq!(x.len(), y.len());
    let len = min(x.len(), y.len());
    unsafe { V::Element::rot()(len, x.as_mut_ptr(), x.stride(), y.as_mut_ptr(), y.stride(), c, s) }
}

/// Setup a Givens rotation with the passed vector `[a, b]`, and returning new elements `(r, z, c, s)`,
/// such that `[[c, s], [-s, c]] * [a, b] = [r, 0]`
#[inline(always)]
pub fn rotg<N>(mut a: N, mut b: N) -> (N, N, N::Float, N) where N: Num {
    let mut s: N = unsafe { std::mem::zeroed() };
    let mut c: N::Float = unsafe { std::mem::zeroed() };
    unsafe { N::rotg()(&mut a as *mut _ as *mut _, &mut b as *mut _ as *mut _, &mut c, &mut s) }
    (a, b, c, s)
}

/// array *= alpha
///
/// *Note:* This crate does not expose zdscal or csscal because they are not actually implemented
/// specially in OpenBLAS (and thus are unlikely to be worthwhile to use).
#[inline(always)]
pub fn scal<V: ?Sized>(alpha: V::Element, x: &mut V) where V: Vector {
    unsafe { V::Element::scal()(x.len(), alpha, x.as_mut_ptr(), x.stride()) }
}

/// Sum of the absolute values of the vector's elements.
///
/// For a complex vector, the "absolute value" is `abs(real) + abs(imag)`
#[inline(always)]
pub fn asum<V: ?Sized>(x: &V) -> <V::Element as Num>::Float where V: Vector {
    unsafe { V::Element::asum()(x.len(), x.as_ptr() as *const _, x.stride()) }
}

/// Index of the first value in the vector with the largest absolute value.
#[inline(always)]
pub fn iamax<V: ?Sized>(x: &V) -> size_t where V: Vector {
    unsafe { V::Element::iamax()(x.len(), x.as_ptr() as *const _, x.stride()) }
}

/// L2 norm of the vector `(sqrt(sum(|x_i|^2)))`, where |x_i| is the complex modulus for a complex number, absolute value otherwise.
#[inline(always)]
pub fn nrm2<V: ?Sized>(x: &V) -> <V::Element as Num>::Float where V: Vector {
    unsafe { V::Element::nrm2()(x.len(), x.as_ptr() as *const _, x.stride()) }
}

/// Do something *really* strange involving a "modified Givens rotation"
pub fn rotm<V: ?Sized, U: ?Sized>(x: &mut V, y: &mut U, param: &[V::Element; 5]) where V: Vector, U: Vector<Element = V::Element>, V::Element: Real {
    debug_assert_eq!(x.len(), y.len());
    let len = min(x.len(), y.len());
    unsafe { V::Element::rotm()(len, x.as_mut_ptr(), x.stride(), y.as_mut_ptr(), y.stride(), param as *const _ as *const _) }
}

/// Setup a "modified Givens rotation", where the parameters and return value are beyond my understanding.
///
/// I think this won't modify the `coord` vector, but I can't really tell from the CBLAS interface
/// whether it does things beyond the Fortran definition.
#[inline(always)]
pub fn rotmg<V: ?Sized, U: ?Sized>(diag: &mut V, coord: &mut U) -> [V::Element; 5] where V: Vector, U: Vector<Element = V::Element>, V::Element: Real {
    // a, b, s COMPLEX, c REAL
    assert!(diag.len() >= 2);
    assert!(coord.len() >= 2);
    debug_assert_eq!(diag.len(), 2);
    debug_assert_eq!(coord.len(), 2);

    let mut param: [V::Element; 5] = unsafe { std::mem::zeroed() };

    unsafe { V::Element::rotmg()(diag.as_mut_ptr() as *mut _, diag.as_mut_ptr().offset(1) as *mut _,
                                 coord.as_mut_ptr() as *mut _, *(coord.as_ptr().offset(1) as *const _),
                                 &mut param as *mut _ as *mut _) }

    param
}

/// Hermitian inner product of the complex vectors x and y
#[inline(always)]
pub fn dotc<V: ?Sized, U: ?Sized>(x: &V, y: &U) -> V::Element where V: Vector, U: Vector<Element = V::Element>, V::Element: Complex {
    debug_assert_eq!(x.len(), y.len());
    let len = min(x.len(), y.len());

    V::Element::from_retself(unsafe { V::Element::dotc()(len, x.as_ptr() as *const _, x.stride(), y.as_ptr() as *const _, y.stride()) })
}

/// General Matrix-vector multiply, y = alpha * A * x + beta * y.
#[inline(always)]
pub fn gemv<V: ?Sized, U: ?Sized, M: ?Sized>(alpha: V::Element, x: &V, beta: V::Element, y: &mut U, a: &M) where V: Vector, U: Vector<Element = V::Element>, M: Matrix<Element = V::Element> {
    debug_assert_eq!(x.len(), a.dim().1);
    let (m, n) = a.dim();
    let len = min(min(x.len(), y.len()), m);

    unsafe { V::Element::gemv()(a.layout() as CBLAS_ORDER, a.transpose() as CBLAS_TRANSPOSE, len, n, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), x.as_ptr() as *const _, x.stride(), beta.as_weird(), y.as_mut_ptr() as *mut _, y.stride()) }
}

/// A = A + alpha * x * y'
#[inline(always)]
pub fn ger<V: ?Sized, U: ?Sized, M: ?Sized>(alpha: V::Element, a: &mut M, x: &V, y: &U) where V: Vector, U: Vector<Element = V::Element>, M: Matrix<Element = V::Element> {
    let (m, n) = a.dim();
    debug_assert_eq!(m, x.len());
    debug_assert_eq!(n, y.len());
    let m = min(m, x.len());
    let n = min(n, y.len());

    unsafe { V::Element::ger()(a.layout() as CBLAS_ORDER, m, n, alpha.as_weird(), x.as_ptr() as *const _, x.stride(), y.as_ptr() as *const _, y.stride(), a.as_mut_ptr() as *mut _, a.major_stride()) }
}

/// Solve the equation A * x = b, storing the result in x.
///
/// A is assumed to be triangular.
#[inline(always)]
pub fn trsv<V: ?Sized, M: ?Sized>(x: &mut V, a: &M) where V: Vector, M: Matrix<Element = V::Element> {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    let len = min(min(m, n), x.len());

    unsafe { V::Element::trsv()(a.layout() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, a.transpose() as CBLAS_TRANSPOSE, a.diag() as CBLAS_DIAG, len, a.as_ptr() as *const _, a.major_stride(), x.as_mut_ptr() as *mut _, x.stride()) }
}

/// Triangular Matrix-vector multiply, x = A * x
#[inline(always)]
pub fn trmv<V: ?Sized, M: ?Sized>(x: &mut V, a: &M) where V: Vector, M: Matrix<Element = V::Element> {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    let len = min(min(m, n), x.len());

    unsafe { V::Element::trmv()(a.layout() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, a.transpose() as CBLAS_TRANSPOSE, a.diag() as CBLAS_DIAG, len, a.as_ptr() as *const _, a.major_stride(), x.as_mut_ptr() as *mut _, x.stride()) }
}

/// General Band Matrix-vector multiply, y = alpha * A * x + beta * y
#[inline(always)]
pub fn gbmv<V: ?Sized, U: ?Sized, M: ?Sized>(alpha: V::Element, x: &V, beta: V::Element, y: &mut U, a: &M) where V: Vector, U: Vector<Element = V::Element>, M: BandMatrix<Element = V::Element> {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), a.dim().0);
    let (m, n) = a.dim();
    let len = min(min(x.len(), y.len()), m);

    unsafe { V::Element::gbmv()(a.layout() as CBLAS_ORDER, a.transpose() as CBLAS_TRANSPOSE, len, n, a.kl(), a.ku(), alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), x.as_ptr() as *const _, x.stride(), beta.as_weird(), y.as_mut_ptr() as *mut _, y.stride()) }
}

/// Triangular Band Matrix-vector multiply, x = A * x
#[inline(always)]
pub fn tbmv<V: ?Sized, M: ?Sized>(x: &mut V, a: &M) where V: Vector, M: BandMatrix<Element = V::Element> {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    let len = min(min(m, n), x.len());

    let k = match a.uplo() {
        Uplo::Lower => a.kl(),
        Uplo::Upper => a.ku(),
    };

    unsafe { V::Element::tbmv()(a.layout() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, a.transpose() as CBLAS_TRANSPOSE, a.diag() as CBLAS_DIAG, len, k, a.as_ptr() as *const _, a.major_stride(), x.as_mut_ptr() as *mut _, x.stride()) }
}

/// Triangular Packed matrix-vector multiply, x = A * x
#[inline(always)]
pub fn tpmv<V: ?Sized, M: ?Sized>(x: &mut V, a: &M) where V: Vector, M: PackedMatrix<Element = V::Element> {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    let len = min(min(m, n), x.len());

    unsafe { V::Element::tpmv()(a.layout() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, a.transpose() as CBLAS_TRANSPOSE, a.diag() as CBLAS_DIAG, len, a.as_ptr() as *const _, x.as_mut_ptr() as *mut _, x.stride()) }
}

/// Solve the equation A * x = b, storing the result in x.
///
/// A is assumed to be triangular and band.
pub fn tbsv<V: ?Sized, M: ?Sized>(x: &mut V, a: &M) where V: Vector, M: BandMatrix<Element = V::Element> {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    let len = min(min(m, n), x.len());

    let k = match a.uplo() {
        Uplo::Lower => a.kl(),
        Uplo::Upper => a.ku(),
    };

    unsafe { V::Element::tbsv()(a.layout() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, a.transpose() as CBLAS_TRANSPOSE, a.diag() as CBLAS_DIAG, len, k, a.as_ptr() as *const _, a.major_stride(), x.as_mut_ptr() as *mut _, x.stride()) }
}

/// Solve the equation A * x = b, storing the result in x.
///
/// A is assumed to be triangular and packed.
#[inline(always)]
pub fn tpsv<V: ?Sized, M: ?Sized>(x: &mut V, a: &M) where V: Vector, M: PackedMatrix<Element = V::Element> {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    let len = min(min(m, n), x.len());

    unsafe { V::Element::tpsv()(a.layout() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, a.transpose() as CBLAS_TRANSPOSE, a.diag() as CBLAS_DIAG, len, a.as_ptr() as *const _, x.as_mut_ptr() as *mut _, x.stride()) }
}

/// A = A + alpha * x * x'
#[inline(always)]
pub fn syr<V: ?Sized, M: ?Sized>(alpha: V::Element, a: &mut M, x: &V) where V: Vector, M: Matrix<Element = V::Element>, V::Element: Real {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    let m = min(m, x.len());

    unsafe { V::Element::syr()(a.layout() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, m, alpha, x.as_ptr() as *const _, x.stride(), a.as_mut_ptr() as *mut _, a.major_stride()) }
}

/// A = A + alpha * x * y'
#[inline(always)]
pub fn syr2<V: ?Sized, U: ?Sized, M: ?Sized>(alpha: V::Element, a: &mut M, x: &V, y: &U) where V: Vector, U: Vector<Element = V::Element>, M: Matrix<Element = V::Element>, V::Element: Real {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    debug_assert_eq!(m, y.len());
    let m = min(m, x.len());

    unsafe { V::Element::syr2()(a.layout() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, m, alpha, x.as_ptr() as *const _, x.stride(), y.as_ptr() as *const _, y.stride(), a.as_mut_ptr() as *mut _, a.major_stride()) }
}

/// Symmetric Matrix-vector multiply, y = alpha * A * x + beta * y
#[inline(always)]
pub fn symv<V: ?Sized, U: ?Sized, M: ?Sized>(alpha: V::Element, x: &V, beta: V::Element, y: &mut U, a: &M) where V: Vector, U: Vector<Element = V::Element>, M: BandMatrix<Element = V::Element>, V::Element: Real {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), m);
    let len = min(min(x.len(), y.len()), m);

    unsafe { V::Element::symv()(a.layout() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, len, alpha, a.as_ptr() as *const _, a.major_stride(), x.as_ptr() as *const _, x.stride(), beta, y.as_mut_ptr() as *mut _, y.stride()) }
}

/// A = A + alpha * x * x'
#[inline(always)]
pub fn spr<V: ?Sized, M: ?Sized>(alpha: V::Element, a: &mut M, x: &V) where V: Vector, M: PackedMatrix<Element = V::Element>, V::Element: Real {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    let m = min(m, x.len());

    unsafe { V::Element::spr()(a.layout() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, m, alpha, x.as_ptr() as *const _, x.stride(), a.as_mut_ptr() as *mut _) }
}

/// A = A + alpha * x * y'
#[inline(always)]
pub fn spr2<V: ?Sized, U: ?Sized, M: ?Sized>(alpha: V::Element, a: &mut M, x: &V, y: &U) where V: Vector, U: Vector<Element = V::Element>, M: PackedMatrix<Element = V::Element>, V::Element: Real {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    debug_assert_eq!(m, y.len());
    let m = min(m, x.len());

    unsafe { V::Element::spr2()(a.layout() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, m, alpha, x.as_ptr() as *const _, x.stride(), y.as_ptr() as *const _, y.stride(), a.as_mut_ptr() as *mut _) }
}

/// Symmetric Packed matrix-vector multiply, y = alpha * A * x + beta * y
#[inline(always)]
pub fn spmv<V: ?Sized, U: ?Sized, M: ?Sized>(alpha: V::Element, x: &V, beta: V::Element, y: &mut U, a: &M) where V: Vector, U: Vector<Element = V::Element>, M: PackedMatrix<Element = V::Element>, V::Element: Real {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), m);
    let len = min(min(x.len(), y.len()), m);

    unsafe { V::Element::spmv()(a.layout() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, len, alpha, a.as_ptr() as *const _, x.as_ptr() as *const _, x.stride(), beta, y.as_mut_ptr() as *mut _, y.stride()) }
}

/// Symetric Band Matrix-vector multiply, y = alpha * A * x + beta * y
#[inline(always)]
pub fn sbmv<V: ?Sized, U: ?Sized, M: ?Sized>(alpha: V::Element, x: &V, beta: V::Element, y: &mut U, a: &M) where V: Vector, U: Vector<Element = V::Element>, M: BandMatrix<Element = V::Element>, V::Element: Real {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    let len = min(min(m, n), x.len());

    let k = match a.uplo() {
        Uplo::Lower => a.kl(),
        Uplo::Upper => a.ku(),
    };

    unsafe { V::Element::sbmv()(a.layout() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, len, k, alpha, a.as_ptr() as *const _, a.major_stride(), x.as_ptr() as *const _, x.stride(), beta, y.as_mut_ptr() as *mut _, y.stride()) }
}

/// A = A + alpha * x * x'
#[inline(always)]
pub fn her<V: ?Sized, M: ?Sized>(alpha: <V::Element as Num>::Float, a: &mut M, x: &V) where V: Vector, M: Matrix<Element = V::Element>, V::Element: Complex {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    let m = min(m, x.len());

    unsafe { V::Element::her()(a.layout() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, m, alpha, x.as_ptr() as *const _, x.stride(), a.as_mut_ptr() as *mut _, a.major_stride()) }
}

/// A = A + alpha * x * y'
#[inline(always)]
pub fn her2<V: ?Sized, U: ?Sized, M: ?Sized>(alpha: V::Element, a: &mut M, x: &V, y: &U) where V: Vector, U: Vector<Element = V::Element>, M: Matrix<Element = V::Element>, V::Element: Complex {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    debug_assert_eq!(m, y.len());
    let m = min(m, x.len());

    unsafe { V::Element::her2()(a.layout() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, m, alpha.as_weird(), x.as_ptr() as *const _, x.stride(), y.as_ptr() as *const _, y.stride(), a.as_mut_ptr() as *mut _, a.major_stride()) }
}

/// Hermitian Matrix-vector multiply, y = alpha * A * x + beta * y
#[inline(always)]
pub fn hemv<V: ?Sized, U: ?Sized, M: ?Sized>(alpha: V::Element, x: &V, beta: V::Element, y: &mut U, a: &M) where V: Vector, U: Vector<Element = V::Element>, M: BandMatrix<Element = V::Element>, V::Element: Complex {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), m);
    let len = min(min(x.len(), y.len()), m);

    unsafe { V::Element::hemv()(a.layout() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, len, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), x.as_ptr() as *const _, x.stride(), beta.as_weird(), y.as_mut_ptr() as *mut _, y.stride()) }
}

/// A = A + alpha * x * x'
#[inline(always)]
pub fn hpr<V: ?Sized, M: ?Sized>(alpha: <V::Element as Num>::Float, a: &mut M, x: &V) where V: Vector, M: PackedMatrix<Element = V::Element>, V::Element: Complex {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    let m = min(m, x.len());

    unsafe { V::Element::hpr()(a.layout() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, m, alpha, x.as_ptr() as *const _, x.stride(), a.as_mut_ptr() as *mut _) }
}

/// A = A + alpha * x * y'
#[inline(always)]
pub fn hpr2<V: ?Sized, U: ?Sized, M: ?Sized>(alpha: V::Element, a: &mut M, x: &V, y: &U) where V: Vector, U: Vector<Element = V::Element>, M: PackedMatrix<Element = V::Element>, V::Element: Complex {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    debug_assert_eq!(m, y.len());
    let m = min(m, x.len());

    unsafe { V::Element::hpr2()(a.layout() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, m, alpha.as_weird(), x.as_ptr() as *const _, x.stride(), y.as_ptr() as *const _, y.stride(), a.as_mut_ptr() as *mut _) }
}

/// Hermitian Packed matrix-vector multiply, y = alpha * A * x + beta * y
#[inline(always)]
pub fn hpmv<V: ?Sized, U: ?Sized, M: ?Sized>(alpha: V::Element, x: &V, beta: V::Element, y: &mut U, a: &M) where V: Vector, U: Vector<Element = V::Element>, M: PackedMatrix<Element = V::Element>, V::Element: Complex {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), m);
    let len = min(min(x.len(), y.len()), m);

    unsafe { V::Element::hpmv()(a.layout() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, len, alpha.as_weird(), a.as_ptr() as *const _, x.as_ptr() as *const _, x.stride(), beta.as_weird(), y.as_mut_ptr() as *mut _, y.stride()) }
}

/// Hermitian Band Matrix-vector multiply, y = alpha * A * x + beta * y
#[inline(always)]
pub fn hbmv<V: ?Sized, U: ?Sized, M: ?Sized>(alpha: V::Element, x: &V, beta: V::Element, y: &mut U, a: &M) where V: Vector, U: Vector<Element = V::Element>, M: BandMatrix<Element = V::Element>, V::Element: Complex {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    let len = min(min(m, n), x.len());

    let k = match a.uplo() {
        Uplo::Lower => a.kl(),
        Uplo::Upper => a.ku(),
    };

    unsafe { V::Element::hbmv()(a.layout() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, len, k, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), x.as_ptr() as *const _, x.stride(), beta.as_weird(), y.as_mut_ptr() as *mut _, y.stride()) }
}

/// General Matrix-matrix multiply, C = alpha * A * B + beta * C
#[inline(always)]
pub unsafe fn gemm<A: ?Sized, B: ?Sized, C: ?Sized>(alpha: A::Element, a: &A, b: &B, beta: A::Element, c: &mut C) where A: Matrix, B: Matrix<Element = A::Element>, C: Matrix<Element = A::Element> {
    let (am, an) = a.dim();
    let (bm, bn) = b.dim();
    let (cm, cn) = c.dim();
    debug_assert_eq!(am, cm);
    debug_assert_eq!(an, bm);
    debug_assert_eq!(bn, cn);

    let m = min(am, cm);
    let n = min(bn, cn);
    let k = min(an, bm);

    unsafe { A::Element::gemm()(a.layout() as CBLAS_ORDER, a.transpose() as CBLAS_TRANSPOSE, b.transpose() as CBLAS_TRANSPOSE, m, n, k, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), b.as_ptr() as *const _, b.major_stride(), beta.as_weird(), c.as_mut_ptr() as *mut _, c.major_stride()) }
}

/// Symetric Matrix-matrix multiply, C = alpha * A * B + beta * C (or, B * A)
///
/// The position of A (to the left or right of B) is controlled by `side`.
#[inline(always)]
pub unsafe fn symm<A: ?Sized, B: ?Sized, C: ?Sized>(side: Side, alpha: A::Element, a: &A, b: &B, beta: A::Element, c: &mut C) where A: Matrix, B: Matrix<Element = A::Element>, C: Matrix<Element = A::Element> {
    let (bm, bn) = b.dim();
    let (cm, cn) = c.dim();
    debug_assert_eq!(bm, cm);
    debug_assert_eq!(bn, cn);

    let m = min(bm, cm);
    let n = min(bn, cn);

    unsafe { A::Element::symm()(a.layout() as CBLAS_ORDER, side as CBLAS_SIDE, c.uplo() as CBLAS_UPLO, m, n, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), b.as_ptr() as *const _, b.major_stride(), beta.as_weird(), c.as_mut_ptr() as *mut _, c.major_stride()) }
}

/// Symetric rank-k operation, C = alpha * A * A' + beta * C,
#[inline(always)]
pub unsafe fn syrk<A: ?Sized, C: ?Sized>(alpha: A::Element, a: &A, beta: A::Element, c: &mut C) where A: Matrix, C: Matrix<Element = A::Element> {
    let (cm, cn) = c.dim();
    debug_assert_eq!(cm, cn);

    let n = cm;
    let k = a.dim().1;

    unsafe { A::Element::syrk()(a.layout() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, a.transpose() as CBLAS_TRANSPOSE, n, k, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), beta.as_weird(), c.as_mut_ptr() as *mut _, c.major_stride()) }
}

/// Symetric rank-2k operation, C = alpha * A * B' + alpha * B * A' + beta * C
#[inline(always)]
pub unsafe fn syr2k<A: ?Sized, B: ?Sized, C: ?Sized>(tran: Transpose, alpha: A::Element, a: &A, b: &B, beta: A::Element, c: &mut C) where A: Matrix, B: Matrix<Element = A::Element>, C: Matrix<Element = A::Element> {
    let (cm, cn) = c.dim();
    debug_assert_eq!(cm, cn);

    let n = cn;
    let k = min(a.dim().1, b.dim().1);

    unsafe { A::Element::syr2k()(a.layout() as CBLAS_ORDER, c.uplo() as CBLAS_UPLO, tran as CBLAS_TRANSPOSE, n, k, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), b.as_ptr() as *const _, b.major_stride(), beta.as_weird(), c.as_mut_ptr() as *mut _, c.major_stride()) }
}

/// Triangular Matrix-matrix multiply, B = alpha * A * B (or, B * A)
///
/// The position of A (to the left or right of B) is controlled by `side`.
#[inline(always)]
pub unsafe fn trmm<A: ?Sized, B: ?Sized>(side: Side, alpha: A::Element, a: &A, b: &mut B) where A: Matrix, B: Matrix<Element = A::Element> {
    let (m, n) = b.dim();

    unsafe { A::Element::trmm()(a.layout() as CBLAS_ORDER, side as CBLAS_SIDE, b.uplo() as CBLAS_UPLO, a.transpose() as CBLAS_TRANSPOSE, a.diag() as CBLAS_DIAG, m, n, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), b.as_mut_ptr() as *mut _, b.major_stride()) }
}

/// Solve the matrix equation A * X = alpha * B (or, X * A)
#[inline(always)]
pub unsafe fn trsm<A: ?Sized, B: ?Sized>(side: Side, alpha: A::Element, a: &A, b: &mut B) where A: Matrix, B: Matrix<Element = A::Element> {
    let (m, n) = b.dim();

    unsafe { A::Element::trsm()(a.layout() as CBLAS_ORDER, side as CBLAS_SIDE, a.uplo() as CBLAS_UPLO, a.transpose() as CBLAS_TRANSPOSE, a.diag() as CBLAS_DIAG, m, n, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), b.as_mut_ptr() as *mut _, b.major_stride()) }
}

/// General complex(?) Matrix-matrix multiply, C = alpha * A * B + beta * C
#[inline(always)]
pub unsafe fn gemm3m<A: ?Sized, B: ?Sized, C: ?Sized>(alpha: A::Element, a: &A, b: &B, beta: A::Element, c: &mut C) where A: Matrix, B: Matrix<Element = A::Element>, C: Matrix<Element = A::Element>, A::Element: Complex {
    let (am, an) = a.dim();
    let (bm, bn) = b.dim();
    let (cm, cn) = c.dim();
    debug_assert_eq!(am, cm);
    debug_assert_eq!(an, bm);
    debug_assert_eq!(bn, cn);

    let m = min(am, cm);
    let n = min(bn, cn);
    let k = min(an, bm);

    unsafe { A::Element::gemm3m()(a.layout() as CBLAS_ORDER, a.transpose() as CBLAS_TRANSPOSE, b.transpose() as CBLAS_TRANSPOSE, m, n, k, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), b.as_ptr() as *const _, b.major_stride(), beta.as_weird(), c.as_mut_ptr() as *mut _, c.major_stride()) }
}

/// Hermitian Matrix-matrix multiply, C = alpha * A * B + beta * C (or, B * A)
///
/// The position of A (to the left or right of B) is controlled by `side`.
#[inline(always)]
pub unsafe fn hemm<A: ?Sized, B: ?Sized, C: ?Sized>(side: Side, alpha: A::Element, a: &A, b: &B, beta: A::Element, c: &mut C) where A: Matrix, B: Matrix<Element = A::Element>, C: Matrix<Element = A::Element>, A::Element: Complex {
    let (bm, bn) = b.dim();
    let (cm, cn) = c.dim();
    debug_assert_eq!(bm, cm);
    debug_assert_eq!(bn, cn);

    let m = min(bm, cm);
    let n = min(bn, cn);

    unsafe { A::Element::hemm()(a.layout() as CBLAS_ORDER, side as CBLAS_SIDE, c.uplo() as CBLAS_UPLO, m, n, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), b.as_ptr() as *const _, b.major_stride(), beta.as_weird(), c.as_mut_ptr() as *mut _, c.major_stride()) }
}

/// Hermitian rank-k operation, C = alpha * A * A' + beta * C,
#[inline(always)]
pub unsafe fn herk<A: ?Sized, C: ?Sized>(alpha: <A::Element as Num>::Float, a: &A, beta: <A::Element as Num>::Float, c: &mut C) where A: Matrix, C: Matrix<Element = A::Element>, C::Element: Complex {
    let (cm, cn) = c.dim();
    debug_assert_eq!(cm, cn);

    let n = cm;
    let k = a.dim().1;

    unsafe { A::Element::herk()(a.layout() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, a.transpose() as CBLAS_TRANSPOSE, n, k, alpha, a.as_ptr() as *const _, a.major_stride(), beta, c.as_mut_ptr() as *mut _, c.major_stride()) }
}

/// Hermitian rank-2k operation, C = alpha * A * B' + alpha * B * A' + beta * C
#[inline(always)]
pub unsafe fn her2k<A: ?Sized, B: ?Sized, C: ?Sized>(tran: Transpose, alpha: A::Element, a: &A, b: &B, beta: <A::Element as Num>::Float, c: &mut C) where A: Matrix, B: Matrix<Element = A::Element>, C: Matrix<Element = A::Element>, A::Element: Complex {
    let (cm, cn) = c.dim();
    debug_assert_eq!(cm, cn);

    let n = cn;
    let k = min(a.dim().1, b.dim().1);

    unsafe { A::Element::her2k()(a.layout() as CBLAS_ORDER, c.uplo() as CBLAS_UPLO, tran as CBLAS_TRANSPOSE, n, k, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), b.as_ptr() as *const _, b.major_stride(), beta, c.as_mut_ptr() as *mut _, c.major_stride()) }
}
