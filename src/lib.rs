//! An interface to the [Basic Linear Algebra Subprograms][1].
//!
//! Note: level 3 (matrix-matrix) functions are marked unsafe, as the matrix traits are not
//! finalized, and I am suspect of their correctness.
//!
//! [1]: http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms

#![allow(unused_unsafe)]

extern crate libblas_sys as raw;
extern crate libc;
extern crate num;

pub mod metal;

type C = num::Complex<f32>;
type Z = num::Complex<f64>;

use raw::*;
use std::mem::transmute;
use std::cmp::min;
use std::ops::{Deref, DerefMut, Add, Sub, Mul, Div};

#[repr(C)]
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum Order {
    RowMajor = CblasRowMajor as isize,
    ColMajor = CblasColMajor as isize,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum Transpose {
    NoTrans = CblasNoTrans as isize,
    Trans = CblasTrans as isize,
    ConjTrans = CblasConjTrans as isize,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum Uplo {
    Upper = CblasUpper as isize,
    Lower = CblasLower as isize,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum Diag {
    NonUnit = CblasNonUnit as isize,
    Unit = CblasUnit as isize,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
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

    fn dot() -> unsafe extern fn(int, *const Self::Float, int, *const Self::Float, int) -> Self::RetSelf;
    fn axpy() -> unsafe extern fn(int, Self, *const Self, int, *mut Self, int);
    fn axpby() -> unsafe extern fn(int, Self, *const Self, int, Self, *mut Self, int);
    fn rot() -> unsafe extern fn(int, *mut Self, int, *mut Self, int, Self::Float, Self);
    fn rotg() -> unsafe extern fn(*mut Self::Float, *mut Self::Float, *mut Self::Float, *mut Self);
    fn scal() -> unsafe extern fn(int, Self, *mut Self, int);
    fn asum() -> unsafe extern fn(int, *const Self::Float, int) -> Self::Float;
    fn iamax() -> unsafe extern fn(int, *const Self::Float, int) -> CBLAS_INDEX;
    fn nrm2() -> unsafe extern fn(int, *const Self::Float, int) -> Self::Float;

    fn gemv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_TRANSPOSE, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, <Self as Num>::Weird, *mut <Self as Num>::Float, int) -> ();
    fn ger() -> unsafe extern fn(CBLAS_ORDER, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> ();
    fn trsv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> ();
    fn trmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> ();
    fn gbmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_TRANSPOSE, int, int, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, <Self as Num>::Weird, *mut <Self as Num>::Float, int) -> ();
    fn tbmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> ();
    fn tpmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, *const <Self as Num>::Float, *mut <Self as Num>::Float, int) -> ();
    fn tbsv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> ();
    fn tpsv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, *const <Self as Num>::Float, *mut <Self as Num>::Float, int) -> ();

    fn gemm() -> unsafe extern fn(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, <Self as Num>::Weird, *mut <Self as Num>::Float, int) -> ();
    fn symm() -> unsafe extern fn(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, <Self as Num>::Weird, *mut <Self as Num>::Float, int) -> ();
    fn syrk() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, <Self as Num>::Weird, *mut <Self as Num>::Float, int) -> ();
    fn syr2k() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, <Self as Num>::Weird, *mut <Self as Num>::Float, int) -> ();
    fn trmm() -> unsafe extern fn(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> ();
    fn trsm() -> unsafe extern fn(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> ();
}

/// A trait representing the various data types BLAS can operate on.
pub unsafe trait Real: Num<Float = Self, RetSelf = Self, Weird = Self> + num::Float {
    fn rotm() -> unsafe extern fn(int, *mut Self, int, *mut Self, int, *const Self);
    fn rotmg() -> unsafe extern fn(*mut Self, *mut Self, *mut Self, Self, *mut Self);

    fn syr() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, Self, *const Self, int, *mut Self, int) -> ();
    fn syr2() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, Self, *const Self, int, *const Self, int, *mut Self, int) -> ();
    fn symv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, Self, *const Self, int, *const Self, int, Self, *mut Self, int) -> ();
    fn spr() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, Self, *const Self, int, *mut Self) -> ();
    fn spr2() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, Self, *const Self, int, *const Self, int, *mut Self) -> ();
    fn spmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, Self, *const Self, *const Self, int, Self, *mut Self, int) -> ();
    fn sbmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, int, Self, *const Self, int, *const Self, int, Self, *mut Self, int) -> ();
}

pub unsafe trait Complex: Num {
    fn dotc() -> unsafe extern fn(int, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int) -> <Self as Num>::RetSelf;
    fn dotu_sub() -> unsafe extern fn(int, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, *mut <Self as Num>::RetSelf);
    fn dotc_sub() -> unsafe extern fn(int, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, *mut <Self as Num>::RetSelf);

    fn her() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, <Self as Num>::Float, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> ();
    fn her2() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> ();
    fn hemv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, <Self as Num>::Weird, *mut <Self as Num>::Float, int) -> ();
    fn hpr() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, <Self as Num>::Float, *const <Self as Num>::Float, int, *mut <Self as Num>::Float) -> ();
    fn hpr2() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, *mut <Self as Num>::Float) -> ();
    fn hbmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, <Self as Num>::Weird, *mut <Self as Num>::Float, int) -> ();
    fn hpmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, <Self as Num>::Weird, *const <Self as Num>::Float, *const <Self as Num>::Float, int, <Self as Num>::Weird, *mut <Self as Num>::Float, int) -> ();

    fn gemm3m() -> unsafe extern fn(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, <Self as Num>::Weird, *mut <Self as Num>::Float, int) -> ();
    fn hemm() -> unsafe extern fn(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, <Self as Num>::Weird, *mut <Self as Num>::Float, int) -> ();
    fn herk() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, <Self as Num>::Float, *const <Self as Num>::Float, int, <Self as Num>::Float, *mut <Self as Num>::Float, int) -> ();
    fn her2k() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, <Self as Num>::Float, *mut <Self as Num>::Float, int) -> ();
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
    fn dot() -> unsafe extern fn(int, *const Self, int, *const Self, int) -> Self { cblas_sdot }
    #[inline(always)]
    fn axpy() -> unsafe extern fn(int, Self, *const Self, int, *mut Self, int) { cblas_saxpy }
    #[inline(always)]
    fn axpby() -> unsafe extern fn(int, Self, *const Self, int, Self, *mut Self, int) { cblas_saxpby }
    #[inline(always)]
    fn rot() -> unsafe extern fn(int, *mut Self, int, *mut Self, int, Self, Self) { cblas_srot }
    #[inline(always)]
    fn rotg() -> unsafe extern fn(*mut Self, *mut Self, *mut Self, *mut Self) { cblas_srotg }
    #[inline(always)]
    fn scal() -> unsafe extern fn(int, Self, *mut Self, int) { cblas_sscal }
    #[inline(always)]
    fn asum() -> unsafe extern fn(int, *const Self, int) -> Self { cblas_sasum }
    #[inline(always)]
    fn iamax() -> unsafe extern fn(int, *const Self, int) -> CBLAS_INDEX { cblas_isamax }
    #[inline(always)]
    fn nrm2() -> unsafe extern fn(int, *const <Self as Num>::Float, int) -> Self { cblas_snrm2 }

    #[inline(always)]
    fn gemv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_TRANSPOSE, int, int, Self, *const Self, int, *const Self, int, Self, *mut Self, int) -> () { cblas_sgemv }
    #[inline(always)]
    fn ger() -> unsafe extern fn(CBLAS_ORDER, int, int, Self, *const Self, int, *const Self, int, *mut Self, int) -> () { cblas_sger }
    #[inline(always)]
    fn trsv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, *const Self, int, *mut Self, int) -> () { cblas_strsv }
    #[inline(always)]
    fn trmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, *const Self, int, *mut Self, int) -> () { cblas_strmv }
    #[inline(always)]
    fn tbsv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, *const Self, int, *mut Self, int) -> () { cblas_stbsv }
    #[inline(always)]
    fn tpsv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, *const Self, *mut Self, int) -> () { cblas_stpsv }
    #[inline(always)]
    fn tpmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, *const Self, *mut Self, int) -> () { cblas_stpmv }
    #[inline(always)]
    fn gbmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_TRANSPOSE, int, int, int, int, Self, *const Self, int, *const Self, int, Self, *mut Self, int) -> () { cblas_sgbmv }
    #[inline(always)]
    fn tbmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, *const Self, int, *mut Self, int) -> () { cblas_stbmv }

    #[inline(always)]
    fn gemm() -> unsafe extern fn(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int, Self, *const Self, int, *const Self, int, Self, *mut Self, int) -> () { cblas_sgemm }
    #[inline(always)]
    fn symm() -> unsafe extern fn(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, int, int, Self, *const Self, int, *const Self, int, Self, *mut Self, int) -> () { cblas_ssymm }
    #[inline(always)]
    fn syrk() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, Self, *const Self, int, Self, *mut Self, int) -> () { cblas_ssyrk }
    #[inline(always)]
    fn syr2k() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, Self, *const Self, int, *const Self, int, Self, *mut Self, int) -> () { cblas_ssyr2k }
    #[inline(always)]
    fn trmm() -> unsafe extern fn(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, Self, *const Self, int, *mut Self, int) -> () { cblas_strmm }
    #[inline(always)]
    fn trsm() -> unsafe extern fn(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, Self, *const Self, int, *mut Self, int) -> () { cblas_strsm }
}

fn c2f(a: &C) -> *const f32 {
    a as *const C as *const f32
}

unsafe extern fn caxpy_wrap(a: i32, b: num::complex::Complex<f32>, c: *const num::complex::Complex<f32>, d: i32, e: *mut num::complex::Complex<f32>, f: i32) {
    cblas_caxpy(a, c2f(&b), c as *const _, d, e as *mut f32, f);
}

unsafe extern fn caxpby_wrap(n: int, alpha: C, x: *const C, incx: int, beta: C, y: *mut C, incy: int) {
    cblas_caxpby(n, c2f(&alpha), x as *const f32, incx, c2f(&beta), y as *mut f32, incy)
}
unsafe extern fn crot_wrap(n: int, x: *mut C, incx: int, y: *mut C, incy: int, c: f32, s: C) {
    cblas_crot(n, x as *mut f32, incx, y as *mut f32, incy, c, c2f(&s));
}
unsafe extern fn cscal_wrap(n: int, alpha: C, x: *mut C, incx: int) {
    cblas_cscal(n, c2f(&alpha), x as *mut f32, incx)
}

unsafe impl Num for C {
    type Float = f32;
    type RetSelf = complex_float;
    type Weird = *const f32;

    #[inline(always)]
    fn as_weird(&self) -> *const f32 { self as *const _ as *const _ }
    #[inline(always)]
    fn from_retself(x: complex_float) -> C { num::Complex { re: x[0], im: x[1] } }

    #[inline(always)]
    fn dot() -> unsafe extern fn(int, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int) -> <Self as Num>::RetSelf { cblas_cdotu }
    #[inline(always)]
    fn axpy() -> unsafe extern fn(int, Self, *const Self, int, *mut Self, int) { caxpy_wrap }
    #[inline(always)]
    fn axpby() -> unsafe extern fn(int, Self, *const Self, int, Self, *mut Self, int) { caxpby_wrap }
    #[inline(always)]
    fn rot() -> unsafe extern fn(int, *mut Self, int, *mut Self, int, <Self as Num>::Float, Self) { crot_wrap }
    #[inline(always)]
    fn rotg() -> unsafe extern fn(*mut <Self as Num>::Float, *mut <Self as Num>::Float, *mut <Self as Num>::Float, *mut Self) { unsafe { transmute(cblas_crotg) } }
    #[inline(always)]
    fn scal() -> unsafe extern fn(int, Self, *mut Self, int) { cscal_wrap }
    #[inline(always)]
    fn asum() -> unsafe extern fn(int, *const <Self as Num>::Float, int) -> <Self as Num>::Float { cblas_scasum }
    #[inline(always)]
    fn iamax() -> unsafe extern fn(int, *const <Self as Num>::Float, int) -> CBLAS_INDEX { cblas_icamax }
    #[inline(always)]
    fn nrm2() -> unsafe extern fn(int, *const <Self as Num>::Float, int) -> <Self as Num>::Float { cblas_scnrm2 }

    #[inline(always)]
    fn gemv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_TRANSPOSE, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, <Self as Num>::Weird, *mut <Self as Num>::Float, int) -> () { cblas_cgemv }
    #[inline(always)]
    fn ger() -> unsafe extern fn(CBLAS_ORDER, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> () { cblas_cgeru }
    #[inline(always)]
    fn trsv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> () { cblas_ctrsv }
    #[inline(always)]
    fn trmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> () { cblas_ctrmv }
    #[inline(always)]
    fn tbsv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> () { cblas_ctbsv }
    #[inline(always)]
    fn tpsv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, *const <Self as Num>::Float, *mut <Self as Num>::Float, int) -> () { cblas_ctpsv }
    #[inline(always)]
    fn tpmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, *const <Self as Num>::Float, *mut <Self as Num>::Float, int) -> () { cblas_ctpmv }
    #[inline(always)]
    fn gbmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_TRANSPOSE, int, int, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, <Self as Num>::Weird, *mut <Self as Num>::Float, int) -> () { cblas_cgbmv }
    #[inline(always)]
    fn tbmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> () { cblas_ctbmv }

    #[inline(always)]
    fn gemm() -> unsafe extern fn(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, <Self as Num>::Weird, *mut <Self as Num>::Float, int) -> () { cblas_cgemm }
    #[inline(always)]
    fn symm() -> unsafe extern fn(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, <Self as Num>::Weird, *mut <Self as Num>::Float, int) -> () { cblas_csymm }
    #[inline(always)]
    fn syrk() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, <Self as Num>::Weird, *mut <Self as Num>::Float, int) -> () { cblas_csyrk }
    #[inline(always)]
    fn syr2k() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, <Self as Num>::Weird, *mut <Self as Num>::Float, int) -> () { cblas_csyr2k }
    #[inline(always)]
    fn trmm() -> unsafe extern fn(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> () { cblas_ctrmm }
    #[inline(always)]
    fn trsm() -> unsafe extern fn(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> () { cblas_ctrsm }
}

unsafe impl Complex for C {
    #[inline(always)]
    fn dotc() -> unsafe extern fn(int, *const <C as Num>::Float, int, *const <C as Num>::Float, int) -> <C as Num>::RetSelf { cblas_cdotc }
    #[inline(always)]
    fn dotu_sub() -> unsafe extern fn(int, *const <C as Num>::Float, int, *const <C as Num>::Float, int, *mut <C as Num>::RetSelf) -> () { cblas_cdotu_sub }
    #[inline(always)]
    fn dotc_sub() -> unsafe extern fn(int, *const <C as Num>::Float, int, *const <C as Num>::Float, int, *mut <C as Num>::RetSelf) -> () { cblas_cdotc_sub }

    #[inline(always)]
    fn her() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, <Self as Num>::Float, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> () { cblas_cher }
    #[inline(always)]
    fn her2() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, *const <Self as Num>::Float, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> () { cblas_cher2 }
    #[inline(always)]
    fn hemv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, *const <Self as Num>::Float, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, *const <Self as Num>::Float, *mut <Self as Num>::Float, int) -> () { cblas_chemv }
    #[inline(always)]
    fn hpr() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, <Self as Num>::Float, *const <Self as Num>::Float, int, *mut <Self as Num>::Float) -> () { cblas_chpr }
    #[inline(always)]
    fn hpr2() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, *const <Self as Num>::Float, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, *mut <Self as Num>::Float) -> () { cblas_chpr2 }
    #[inline(always)]
    fn hbmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, int, *const <Self as Num>::Float, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, *const <Self as Num>::Float, *mut <Self as Num>::Float, int) -> () { cblas_chbmv }
    #[inline(always)]
    fn hpmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, *const <Self as Num>::Float, *const <Self as Num>::Float, *const <Self as Num>::Float, int, *const <Self as Num>::Float, *mut <Self as Num>::Float, int) -> () { cblas_chpmv }

    #[inline(always)]
    fn gemm3m() -> unsafe extern fn(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int, *const <Self as Num>::Float, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, *const <Self as Num>::Float, *mut <Self as Num>::Float, int) -> () { cblas_cgemm3m }
    #[inline(always)]
    fn hemm() -> unsafe extern fn(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, int, int, *const <Self as Num>::Float, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, *const <Self as Num>::Float, *mut <Self as Num>::Float, int) -> () { cblas_chemm }
    #[inline(always)]
    fn herk() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, <Self as Num>::Float, *const <Self as Num>::Float, int, <Self as Num>::Float, *mut <Self as Num>::Float, int) -> () { cblas_cherk }
    #[inline(always)]
    fn her2k() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, *const <Self as Num>::Float, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, <Self as Num>::Float, *mut <Self as Num>::Float, int) -> () { cblas_cher2k }
}

fn z2f(a: &Z) -> *const f64 {
    a as *const Z as *const f64
}

unsafe extern fn zaxpy_wrap(a: i32, b: Z, c: *const Z, d: i32, e: *mut Z, f: i32) {
    cblas_zaxpy(a, z2f(&b), c as *const f64, d, e as *mut f64, f);
}

unsafe extern fn zaxpby_wrap(n: int, alpha: Z, x: *const Z, incx: int, beta: Z, y: *mut Z, incy: int) {
    cblas_zaxpby(n, z2f(&alpha), x as *const f64, incx, z2f(&beta), y as *mut f64, incy)
}

unsafe extern fn zrot_wrap(n: int, x: *mut Z, incx: int, y: *mut Z, incy: int, c: f64, s: Z) {
    cblas_zrot(n, x as *mut f64, incx, y as *mut f64, incy, c, z2f(&s));
}

unsafe extern fn zscal_wrap(n: int, alpha: Z, x: *mut Z, incx: int) {
    cblas_zscal(n, z2f(&alpha), x as *mut f64, incx)
}

unsafe impl Num for Z {
    type Float = f64;
    type RetSelf = complex_double;
    type Weird = *const f64;

    #[inline(always)]
    fn as_weird(&self) -> *const f64 { self as *const _ as *const _ }
    #[inline(always)]
    fn from_retself(x: complex_double) -> Z { num::Complex { re: x[0], im: x[1] } }

    #[inline(always)]
    fn dot() -> unsafe extern fn(int, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int) -> <Self as Num>::RetSelf { cblas_zdotu }
    #[inline(always)]
    fn axpy() -> unsafe extern fn(int, Self, *const Self, int, *mut Self, int) { zaxpy_wrap }
    #[inline(always)]
    fn axpby() -> unsafe extern fn(int, Self, *const Self, int, Self, *mut Self, int) { zaxpby_wrap }
    #[inline(always)]
    fn rot() -> unsafe extern fn(int, *mut Self, int, *mut Self, int, <Self as Num>::Float, Self) { zrot_wrap }
    #[inline(always)]
    fn rotg() -> unsafe extern fn(*mut <Self as Num>::Float, *mut <Self as Num>::Float, *mut <Self as Num>::Float, *mut Self) { unsafe { transmute(cblas_zrotg) } }
    #[inline(always)]
    fn scal() -> unsafe extern fn(int, Self, *mut Self, int) { zscal_wrap }
    #[inline(always)]
    fn asum() -> unsafe extern fn(int, *const <Self as Num>::Float, int) -> <Self as Num>::Float { cblas_dzasum }
    #[inline(always)]
    fn iamax() -> unsafe extern fn(int, *const <Self as Num>::Float, int) -> CBLAS_INDEX { cblas_izamax }
    #[inline(always)]
    fn nrm2() -> unsafe extern fn(int, *const <Self as Num>::Float, int) -> <Self as Num>::Float { cblas_dznrm2 }

    #[inline(always)]
    fn gemv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_TRANSPOSE, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, <Self as Num>::Weird, *mut <Self as Num>::Float, int) -> () { cblas_zgemv }
    #[inline(always)]
    fn ger() -> unsafe extern fn(CBLAS_ORDER, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> () { cblas_zgeru }
    #[inline(always)]
    fn trsv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> () { cblas_ztrsv }
    #[inline(always)]
    fn trmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> () { cblas_ztrmv }
    #[inline(always)]
    fn tbsv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> () { cblas_ztbsv }
    #[inline(always)]
    fn tpsv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, *const <Self as Num>::Float, *mut <Self as Num>::Float, int) -> () { cblas_ztpsv }
    #[inline(always)]
    fn tpmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, *const <Self as Num>::Float, *mut <Self as Num>::Float, int) -> () { cblas_ztpmv }
    #[inline(always)]
    fn gbmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_TRANSPOSE, int, int, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, <Self as Num>::Weird, *mut <Self as Num>::Float, int) -> () { cblas_zgbmv }
    #[inline(always)]
    fn tbmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> () { cblas_ztbmv }

    #[inline(always)]
    fn gemm() -> unsafe extern fn(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, <Self as Num>::Weird, *mut <Self as Num>::Float, int) -> () { cblas_zgemm }
    #[inline(always)]
    fn symm() -> unsafe extern fn(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, <Self as Num>::Weird, *mut <Self as Num>::Float, int) -> () { cblas_zsymm }
    #[inline(always)]
    fn syrk() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, <Self as Num>::Weird, *mut <Self as Num>::Float, int) -> () { cblas_zsyrk }
    #[inline(always)]
    fn syr2k() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, <Self as Num>::Weird, *mut <Self as Num>::Float, int) -> () { cblas_zsyr2k }
    #[inline(always)]
    fn trmm() -> unsafe extern fn(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> () { cblas_ztrmm }
    #[inline(always)]
    fn trsm() -> unsafe extern fn(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, <Self as Num>::Weird, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> () { cblas_ztrsm }
}

unsafe impl Complex for Z {
    #[inline(always)]
    fn dotc() -> unsafe extern fn(int, *const <Z as Num>::Float, int, *const <Z as Num>::Float, int) -> <Z as Num>::RetSelf { cblas_zdotc }
    #[inline(always)]
    fn dotu_sub() -> unsafe extern fn(int, *const <Z as Num>::Float, int, *const <Z as Num>::Float, int, *mut <Z as Num>::RetSelf) -> () { cblas_zdotu_sub }
    #[inline(always)]
    fn dotc_sub() -> unsafe extern fn(int, *const <Z as Num>::Float, int, *const <Z as Num>::Float, int, *mut <Z as Num>::RetSelf) -> () { cblas_zdotc_sub }

    #[inline(always)]
    fn her() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, <Self as Num>::Float, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> () { cblas_zher }
    #[inline(always)]
    fn her2() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, *const <Self as Num>::Float, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, *mut <Self as Num>::Float, int) -> () { cblas_zher2 }
    #[inline(always)]
    fn hemv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, *const <Self as Num>::Float, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, *const <Self as Num>::Float, *mut <Self as Num>::Float, int) -> () { cblas_zhemv }
    #[inline(always)]
    fn hpr() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, <Self as Num>::Float, *const <Self as Num>::Float, int, *mut <Self as Num>::Float) -> () { cblas_zhpr }
    #[inline(always)]
    fn hpr2() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, *const <Self as Num>::Float, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, *mut <Self as Num>::Float) -> () { cblas_zhpr2 }
    #[inline(always)]
    fn hbmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, int, *const <Self as Num>::Float, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, *const <Self as Num>::Float, *mut <Self as Num>::Float, int) -> () { cblas_zhbmv }
    #[inline(always)]
    fn hpmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, *const <Self as Num>::Float, *const <Self as Num>::Float, *const <Self as Num>::Float, int, *const <Self as Num>::Float, *mut <Self as Num>::Float, int) -> () { cblas_zhpmv }

    #[inline(always)]
    fn gemm3m() -> unsafe extern fn(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int, *const <Self as Num>::Float, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, *const <Self as Num>::Float, *mut <Self as Num>::Float, int) -> () { cblas_zgemm3m }
    #[inline(always)]
    fn hemm() -> unsafe extern fn(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, int, int, *const <Self as Num>::Float, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, *const <Self as Num>::Float, *mut <Self as Num>::Float, int) -> () { cblas_zhemm }
    #[inline(always)]
    fn herk() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, <Self as Num>::Float, *const <Self as Num>::Float, int, <Self as Num>::Float, *mut <Self as Num>::Float, int) -> () { cblas_zherk }
    #[inline(always)]
    fn her2k() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, *const <Self as Num>::Float, *const <Self as Num>::Float, int, *const <Self as Num>::Float, int, <Self as Num>::Float, *mut <Self as Num>::Float, int) -> () { cblas_zher2k }
}

unsafe impl Real for f32 {
    #[inline(always)]
    fn rotm() -> unsafe extern fn(int, *mut Self, int, *mut Self, int, *const Self) { cblas_srotm }
    #[inline(always)]
    fn rotmg() -> unsafe extern fn( *mut Self, *mut Self, *mut Self, Self, *mut Self) { cblas_srotmg }

    #[inline(always)]
    fn syr() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, Self, *const Self, int, *mut Self, int) -> () { cblas_ssyr }
    #[inline(always)]
    fn syr2() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, Self, *const Self, int, *const Self, int, *mut Self, int) -> () { cblas_ssyr2 }
    #[inline(always)]
    fn symv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, Self, *const Self, int, *const Self, int, Self, *mut Self, int) -> () { cblas_ssymv }
    #[inline(always)]
    fn spr() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, Self, *const Self, int, *mut Self) -> () { cblas_sspr }
    #[inline(always)]
    fn spr2() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, Self, *const Self, int, *const Self, int, *mut Self) -> () { cblas_sspr2 }
    #[inline(always)]
    fn spmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, Self, *const Self, *const Self, int, Self, *mut Self, int) -> () { cblas_sspmv }
    #[inline(always)]
    fn sbmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, int, Self, *const Self, int, *const Self, int, Self, *mut Self, int) -> () { cblas_ssbmv }
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
    fn dot() -> unsafe extern fn(int, *const Self, int, *const Self, int) -> Self { cblas_ddot }
    #[inline(always)]
    fn axpy() -> unsafe extern fn(int, Self, *const Self, int, *mut Self, int) { cblas_daxpy }
    #[inline(always)]
    fn axpby() -> unsafe extern fn(int, Self, *const Self, int, Self, *mut Self, int) { cblas_daxpby }
    #[inline(always)]
    fn rot() -> unsafe extern fn(int, *mut Self, int, *mut Self, int, Self, Self) { cblas_drot }
    #[inline(always)]
    fn rotg() -> unsafe extern fn(*mut Self, *mut Self, *mut Self, *mut Self) { cblas_drotg }
    #[inline(always)]
    fn scal() -> unsafe extern fn(int, Self, *mut Self, int) { cblas_dscal }
    #[inline(always)]
    fn asum() -> unsafe extern fn(int, *const Self, int) -> Self { cblas_dasum }
    #[inline(always)]
    fn iamax() -> unsafe extern fn(int, *const Self, int) -> CBLAS_INDEX { cblas_idamax }
    #[inline(always)]
    fn nrm2() -> unsafe extern fn(int, *const Self, int) -> Self { cblas_dnrm2 }

    #[inline(always)]
    fn gemv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_TRANSPOSE, int, int, Self, *const Self, int, *const Self, int, Self, *mut Self, int) -> () { cblas_dgemv }
    #[inline(always)]
    fn ger() -> unsafe extern fn(CBLAS_ORDER, int, int, Self, *const Self, int, *const Self, int, *mut Self, int) -> () { cblas_dger }
    #[inline(always)]
    fn trsv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, *const Self, int, *mut Self, int) -> () { cblas_dtrsv }
    #[inline(always)]
    fn trmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, *const Self, int, *mut Self, int) -> () { cblas_dtrmv }
    #[inline(always)]
    fn tbsv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, *const Self, int, *mut Self, int) -> () { cblas_dtbsv }
    #[inline(always)]
    fn tpsv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, *const Self, *mut Self, int) -> () { cblas_dtpsv }
    #[inline(always)]
    fn tpmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, *const Self, *mut Self, int) -> () { cblas_dtpmv }
    #[inline(always)]
    fn gbmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_TRANSPOSE, int, int, int, int, Self, *const Self, int, *const Self, int, Self, *mut Self, int) -> () { cblas_dgbmv }
    #[inline(always)]
    fn tbmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, *const Self, int, *mut Self, int) -> () { cblas_dtbmv }

    #[inline(always)]
    fn gemm() -> unsafe extern fn(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int, Self, *const Self, int, *const Self, int, Self, *mut Self, int) -> () { cblas_dgemm }
    #[inline(always)]
    fn symm() -> unsafe extern fn(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, int, int, Self, *const Self, int, *const Self, int, Self, *mut Self, int) -> () { cblas_dsymm }
    #[inline(always)]
    fn syrk() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, Self, *const Self, int, Self, *mut Self, int) -> () { cblas_dsyrk }
    #[inline(always)]
    fn syr2k() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, Self, *const Self, int, *const Self, int, Self, *mut Self, int) -> () { cblas_dsyr2k }
    #[inline(always)]
    fn trmm() -> unsafe extern fn(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, Self, *const Self, int, *mut Self, int) -> () { cblas_dtrmm }
    #[inline(always)]
    fn trsm() -> unsafe extern fn(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, Self, *const Self, int, *mut Self, int) -> () { cblas_dtrsm }
}

unsafe impl Real for f64 {
    #[inline(always)]
    fn rotm() -> unsafe extern fn(int, *mut Self, int, *mut Self, int, *const Self) { cblas_drotm }
    #[inline(always)]
    fn rotmg() -> unsafe extern fn(*mut Self, *mut Self, *mut Self, Self, *mut Self) { cblas_drotmg }

    #[inline(always)]
    fn syr() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, Self, *const Self, int, *mut Self, int) -> () { cblas_dsyr }
    #[inline(always)]
    fn syr2() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, Self, *const Self, int, *const Self, int, *mut Self, int) -> () { cblas_dsyr2 }
    #[inline(always)]
    fn symv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, Self, *const Self, int, *const Self, int, Self, *mut Self, int) -> () { cblas_dsymv }
    #[inline(always)]
    fn spr() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, Self, *const Self, int, *mut Self) -> () { cblas_dspr }
    #[inline(always)]
    fn spr2() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, Self, *const Self, int, *const Self, int, *mut Self) -> () { cblas_dspr2 }
    #[inline(always)]
    fn spmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, Self, *const Self, *const Self, int, Self, *mut Self, int) -> () { cblas_dspmv }
    #[inline(always)]
    fn sbmv() -> unsafe extern fn(CBLAS_ORDER, CBLAS_UPLO, int, int, Self, *const Self, int, *const Self, int, Self, *mut Self, int) -> () { cblas_dsbmv }
}

pub unsafe trait Vector {
    type Element: Num;

    /// Number of elements in the vector.
    fn len(&self) -> int;

    /// The number of elements between consecutive vector entries.
    ///
    /// This is *not* in bytes!
    fn stride(&self) -> int;

    fn as_ptr(&self) -> *const Self::Element;
    fn as_mut_ptr(&mut self) -> *mut Self::Element;
}

unsafe impl<T: Num> Vector for [T] {
    type Element = T;

    #[inline(always)]
    fn len(&self) -> int {
        <[T]>::len(self) as int
    }

    #[inline(always)]
    fn stride(&self) -> int {
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
    fn dim(&self) -> (int, int);

    /// The "stride" of the major ("leading") dimension.
    ///
    /// If the matrix is row-major, then this is the number of elements between entries in adjacent
    /// rows with the same column index. This can be useful when "slicing" a portion of a matrix,
    /// but is usually just going to be the number of rows/columns.
    fn major_stride(&self) -> int {
        let (m, n) = self.dim();
        match self.order() {
            Order::RowMajor => n,
            Order::ColMajor => m,
        }
    }

    fn order(&self) -> Order { Order::RowMajor }
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
    fn kl(&self) -> int;
    /// Number of super-diagonals.
    fn ku(&self) -> int;
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
    rows: int,
    cols: int,
    data: Vec<T>,
    pub tran: Transpose,
    pub uplo: Uplo,
    pub diag: Diag,
}

impl<T> VecMatrix<T> {
    /// Create a non-transposed, upper, non-unit matrix.
    pub fn from_parts(rows: int, cols: int, data: Vec<T>) -> VecMatrix<T> {
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

    fn dim(&self) -> (int, int) {
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
pub fn iamax<V: ?Sized>(x: &V) -> usize where V: Vector {
    unsafe { V::Element::iamax()(x.len(), x.as_ptr() as *const _, x.stride()) as usize }
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

    unsafe { V::Element::gemv()(a.order() as CBLAS_ORDER, a.transpose() as CBLAS_TRANSPOSE, len, n, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), x.as_ptr() as *const _, x.stride(), beta.as_weird(), y.as_mut_ptr() as *mut _, y.stride()) }
}

/// A = A + alpha * x * y'
#[inline(always)]
pub fn ger<V: ?Sized, U: ?Sized, M: ?Sized>(alpha: V::Element, a: &mut M, x: &V, y: &U) where V: Vector, U: Vector<Element = V::Element>, M: Matrix<Element = V::Element> {
    let (m, n) = a.dim();
    debug_assert_eq!(m, x.len());
    debug_assert_eq!(n, y.len());
    let m = min(m, x.len());
    let n = min(n, y.len());

    unsafe { V::Element::ger()(a.order() as CBLAS_ORDER, m, n, alpha.as_weird(), x.as_ptr() as *const _, x.stride(), y.as_ptr() as *const _, y.stride(), a.as_mut_ptr() as *mut _, a.major_stride()) }
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

    unsafe { V::Element::trsv()(a.order() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, a.transpose() as CBLAS_TRANSPOSE, a.diag() as CBLAS_DIAG, len, a.as_ptr() as *const _, a.major_stride(), x.as_mut_ptr() as *mut _, x.stride()) }
}

/// Triangular Matrix-vector multiply, x = A * x
#[inline(always)]
pub fn trmv<V: ?Sized, M: ?Sized>(x: &mut V, a: &M) where V: Vector, M: Matrix<Element = V::Element> {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    let len = min(min(m, n), x.len());

    unsafe { V::Element::trmv()(a.order() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, a.transpose() as CBLAS_TRANSPOSE, a.diag() as CBLAS_DIAG, len, a.as_ptr() as *const _, a.major_stride(), x.as_mut_ptr() as *mut _, x.stride()) }
}

/// General Band Matrix-vector multiply, y = alpha * A * x + beta * y
#[inline(always)]
pub fn gbmv<V: ?Sized, U: ?Sized, M: ?Sized>(alpha: V::Element, x: &V, beta: V::Element, y: &mut U, a: &M) where V: Vector, U: Vector<Element = V::Element>, M: BandMatrix<Element = V::Element> {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), a.dim().0);
    let (m, n) = a.dim();
    let len = min(min(x.len(), y.len()), m);

    unsafe { V::Element::gbmv()(a.order() as CBLAS_ORDER, a.transpose() as CBLAS_TRANSPOSE, len, n, a.kl(), a.ku(), alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), x.as_ptr() as *const _, x.stride(), beta.as_weird(), y.as_mut_ptr() as *mut _, y.stride()) }
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

    unsafe { V::Element::tbmv()(a.order() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, a.transpose() as CBLAS_TRANSPOSE, a.diag() as CBLAS_DIAG, len, k, a.as_ptr() as *const _, a.major_stride(), x.as_mut_ptr() as *mut _, x.stride()) }
}

/// Triangular Packed matrix-vector multiply, x = A * x
#[inline(always)]
pub fn tpmv<V: ?Sized, M: ?Sized>(x: &mut V, a: &M) where V: Vector, M: PackedMatrix<Element = V::Element> {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    let len = min(min(m, n), x.len());

    unsafe { V::Element::tpmv()(a.order() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, a.transpose() as CBLAS_TRANSPOSE, a.diag() as CBLAS_DIAG, len, a.as_ptr() as *const _, x.as_mut_ptr() as *mut _, x.stride()) }
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

    unsafe { V::Element::tbsv()(a.order() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, a.transpose() as CBLAS_TRANSPOSE, a.diag() as CBLAS_DIAG, len, k, a.as_ptr() as *const _, a.major_stride(), x.as_mut_ptr() as *mut _, x.stride()) }
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

    unsafe { V::Element::tpsv()(a.order() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, a.transpose() as CBLAS_TRANSPOSE, a.diag() as CBLAS_DIAG, len, a.as_ptr() as *const _, x.as_mut_ptr() as *mut _, x.stride()) }
}

/// A = A + alpha * x * x'
#[inline(always)]
pub fn syr<V: ?Sized, M: ?Sized>(alpha: V::Element, a: &mut M, x: &V) where V: Vector, M: Matrix<Element = V::Element>, V::Element: Real {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    let m = min(m, x.len());

    unsafe { V::Element::syr()(a.order() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, m, alpha, x.as_ptr() as *const _, x.stride(), a.as_mut_ptr() as *mut _, a.major_stride()) }
}

/// A = A + alpha * x * y'
#[inline(always)]
pub fn syr2<V: ?Sized, U: ?Sized, M: ?Sized>(alpha: V::Element, a: &mut M, x: &V, y: &U) where V: Vector, U: Vector<Element = V::Element>, M: Matrix<Element = V::Element>, V::Element: Real {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    debug_assert_eq!(m, y.len());
    let m = min(m, x.len());

    unsafe { V::Element::syr2()(a.order() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, m, alpha, x.as_ptr() as *const _, x.stride(), y.as_ptr() as *const _, y.stride(), a.as_mut_ptr() as *mut _, a.major_stride()) }
}

/// Symmetric Matrix-vector multiply, y = alpha * A * x + beta * y
#[inline(always)]
pub fn symv<V: ?Sized, U: ?Sized, M: ?Sized>(alpha: V::Element, x: &V, beta: V::Element, y: &mut U, a: &M) where V: Vector, U: Vector<Element = V::Element>, M: BandMatrix<Element = V::Element>, V::Element: Real {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), m);
    let len = min(min(x.len(), y.len()), m);

    unsafe { V::Element::symv()(a.order() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, len, alpha, a.as_ptr() as *const _, a.major_stride(), x.as_ptr() as *const _, x.stride(), beta, y.as_mut_ptr() as *mut _, y.stride()) }
}

/// A = A + alpha * x * x'
#[inline(always)]
pub fn spr<V: ?Sized, M: ?Sized>(alpha: V::Element, a: &mut M, x: &V) where V: Vector, M: PackedMatrix<Element = V::Element>, V::Element: Real {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    let m = min(m, x.len());

    unsafe { V::Element::spr()(a.order() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, m, alpha, x.as_ptr() as *const _, x.stride(), a.as_mut_ptr() as *mut _) }
}

/// A = A + alpha * x * y'
#[inline(always)]
pub fn spr2<V: ?Sized, U: ?Sized, M: ?Sized>(alpha: V::Element, a: &mut M, x: &V, y: &U) where V: Vector, U: Vector<Element = V::Element>, M: PackedMatrix<Element = V::Element>, V::Element: Real {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    debug_assert_eq!(m, y.len());
    let m = min(m, x.len());

    unsafe { V::Element::spr2()(a.order() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, m, alpha, x.as_ptr() as *const _, x.stride(), y.as_ptr() as *const _, y.stride(), a.as_mut_ptr() as *mut _) }
}

/// Symmetric Packed matrix-vector multiply, y = alpha * A * x + beta * y
#[inline(always)]
pub fn spmv<V: ?Sized, U: ?Sized, M: ?Sized>(alpha: V::Element, x: &V, beta: V::Element, y: &mut U, a: &M) where V: Vector, U: Vector<Element = V::Element>, M: PackedMatrix<Element = V::Element>, V::Element: Real {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), m);
    let len = min(min(x.len(), y.len()), m);

    unsafe { V::Element::spmv()(a.order() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, len, alpha, a.as_ptr() as *const _, x.as_ptr() as *const _, x.stride(), beta, y.as_mut_ptr() as *mut _, y.stride()) }
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

    unsafe { V::Element::sbmv()(a.order() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, len, k, alpha, a.as_ptr() as *const _, a.major_stride(), x.as_ptr() as *const _, x.stride(), beta, y.as_mut_ptr() as *mut _, y.stride()) }
}

/// A = A + alpha * x * x'
#[inline(always)]
pub fn her<V: ?Sized, M: ?Sized>(alpha: <V::Element as Num>::Float, a: &mut M, x: &V) where V: Vector, M: Matrix<Element = V::Element>, V::Element: Complex {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    let m = min(m, x.len());

    unsafe { V::Element::her()(a.order() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, m, alpha, x.as_ptr() as *const _, x.stride(), a.as_mut_ptr() as *mut _, a.major_stride()) }
}

/// A = A + alpha * x * y'
#[inline(always)]
pub fn her2<V: ?Sized, U: ?Sized, M: ?Sized>(alpha: V::Element, a: &mut M, x: &V, y: &U) where V: Vector, U: Vector<Element = V::Element>, M: Matrix<Element = V::Element>, V::Element: Complex {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    debug_assert_eq!(m, y.len());
    let m = min(m, x.len());

    unsafe { V::Element::her2()(a.order() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, m, alpha.as_weird(), x.as_ptr() as *const _, x.stride(), y.as_ptr() as *const _, y.stride(), a.as_mut_ptr() as *mut _, a.major_stride()) }
}

/// Hermitian Matrix-vector multiply, y = alpha * A * x + beta * y
#[inline(always)]
pub fn hemv<V: ?Sized, U: ?Sized, M: ?Sized>(alpha: V::Element, x: &V, beta: V::Element, y: &mut U, a: &M) where V: Vector, U: Vector<Element = V::Element>, M: BandMatrix<Element = V::Element>, V::Element: Complex {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), m);
    let len = min(min(x.len(), y.len()), m);

    unsafe { V::Element::hemv()(a.order() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, len, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), x.as_ptr() as *const _, x.stride(), beta.as_weird(), y.as_mut_ptr() as *mut _, y.stride()) }
}

/// A = A + alpha * x * x'
#[inline(always)]
pub fn hpr<V: ?Sized, M: ?Sized>(alpha: <V::Element as Num>::Float, a: &mut M, x: &V) where V: Vector, M: PackedMatrix<Element = V::Element>, V::Element: Complex {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    let m = min(m, x.len());

    unsafe { V::Element::hpr()(a.order() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, m, alpha, x.as_ptr() as *const _, x.stride(), a.as_mut_ptr() as *mut _) }
}

/// A = A + alpha * x * y'
#[inline(always)]
pub fn hpr2<V: ?Sized, U: ?Sized, M: ?Sized>(alpha: V::Element, a: &mut M, x: &V, y: &U) where V: Vector, U: Vector<Element = V::Element>, M: PackedMatrix<Element = V::Element>, V::Element: Complex {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(m, x.len());
    debug_assert_eq!(m, y.len());
    let m = min(m, x.len());

    unsafe { V::Element::hpr2()(a.order() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, m, alpha.as_weird(), x.as_ptr() as *const _, x.stride(), y.as_ptr() as *const _, y.stride(), a.as_mut_ptr() as *mut _) }
}

/// Hermitian Packed matrix-vector multiply, y = alpha * A * x + beta * y
#[inline(always)]
pub fn hpmv<V: ?Sized, U: ?Sized, M: ?Sized>(alpha: V::Element, x: &V, beta: V::Element, y: &mut U, a: &M) where V: Vector, U: Vector<Element = V::Element>, M: PackedMatrix<Element = V::Element>, V::Element: Complex {
    let (m, n) = a.dim();
    debug_assert_eq!(m, n);
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), m);
    let len = min(min(x.len(), y.len()), m);

    unsafe { V::Element::hpmv()(a.order() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, len, alpha.as_weird(), a.as_ptr() as *const _, x.as_ptr() as *const _, x.stride(), beta.as_weird(), y.as_mut_ptr() as *mut _, y.stride()) }
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

    unsafe { V::Element::hbmv()(a.order() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, len, k, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), x.as_ptr() as *const _, x.stride(), beta.as_weird(), y.as_mut_ptr() as *mut _, y.stride()) }
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

    unsafe { A::Element::gemm()(a.order() as CBLAS_ORDER, a.transpose() as CBLAS_TRANSPOSE, b.transpose() as CBLAS_TRANSPOSE, m, n, k, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), b.as_ptr() as *const _, b.major_stride(), beta.as_weird(), c.as_mut_ptr() as *mut _, c.major_stride()) }
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

    unsafe { A::Element::symm()(a.order() as CBLAS_ORDER, side as CBLAS_SIDE, c.uplo() as CBLAS_UPLO, m, n, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), b.as_ptr() as *const _, b.major_stride(), beta.as_weird(), c.as_mut_ptr() as *mut _, c.major_stride()) }
}

/// Symetric rank-k operation, C = alpha * A * A' + beta * C,
#[inline(always)]
pub unsafe fn syrk<A: ?Sized, C: ?Sized>(alpha: A::Element, a: &A, beta: A::Element, c: &mut C) where A: Matrix, C: Matrix<Element = A::Element> {
    let (cm, cn) = c.dim();
    debug_assert_eq!(cm, cn);

    let n = cm;
    let k = a.dim().1;

    unsafe { A::Element::syrk()(a.order() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, a.transpose() as CBLAS_TRANSPOSE, n, k, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), beta.as_weird(), c.as_mut_ptr() as *mut _, c.major_stride()) }
}

/// Symetric rank-2k operation, C = alpha * A * B' + alpha * B * A' + beta * C
#[inline(always)]
pub unsafe fn syr2k<A: ?Sized, B: ?Sized, C: ?Sized>(tran: Transpose, alpha: A::Element, a: &A, b: &B, beta: A::Element, c: &mut C) where A: Matrix, B: Matrix<Element = A::Element>, C: Matrix<Element = A::Element> {
    let (cm, cn) = c.dim();
    debug_assert_eq!(cm, cn);

    let n = cn;
    let k = min(a.dim().1, b.dim().1);

    unsafe { A::Element::syr2k()(a.order() as CBLAS_ORDER, c.uplo() as CBLAS_UPLO, tran as CBLAS_TRANSPOSE, n, k, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), b.as_ptr() as *const _, b.major_stride(), beta.as_weird(), c.as_mut_ptr() as *mut _, c.major_stride()) }
}

/// Triangular Matrix-matrix multiply, B = alpha * A * B (or, B * A)
///
/// The position of A (to the left or right of B) is controlled by `side`.
#[inline(always)]
pub unsafe fn trmm<A: ?Sized, B: ?Sized>(side: Side, alpha: A::Element, a: &A, b: &mut B) where A: Matrix, B: Matrix<Element = A::Element> {
    let (m, n) = b.dim();

    unsafe { A::Element::trmm()(a.order() as CBLAS_ORDER, side as CBLAS_SIDE, b.uplo() as CBLAS_UPLO, a.transpose() as CBLAS_TRANSPOSE, a.diag() as CBLAS_DIAG, m, n, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), b.as_mut_ptr() as *mut _, b.major_stride()) }
}

/// Solve the matrix equation A * X = alpha * B (or, X * A)
#[inline(always)]
pub unsafe fn trsm<A: ?Sized, B: ?Sized>(side: Side, alpha: A::Element, a: &A, b: &mut B) where A: Matrix, B: Matrix<Element = A::Element> {
    let (m, n) = b.dim();

    unsafe { A::Element::trsm()(a.order() as CBLAS_ORDER, side as CBLAS_SIDE, a.uplo() as CBLAS_UPLO, a.transpose() as CBLAS_TRANSPOSE, a.diag() as CBLAS_DIAG, m, n, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), b.as_mut_ptr() as *mut _, b.major_stride()) }
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

    unsafe { A::Element::gemm3m()(a.order() as CBLAS_ORDER, a.transpose() as CBLAS_TRANSPOSE, b.transpose() as CBLAS_TRANSPOSE, m, n, k, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), b.as_ptr() as *const _, b.major_stride(), beta.as_weird(), c.as_mut_ptr() as *mut _, c.major_stride()) }
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

    unsafe { A::Element::hemm()(a.order() as CBLAS_ORDER, side as CBLAS_SIDE, c.uplo() as CBLAS_UPLO, m, n, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), b.as_ptr() as *const _, b.major_stride(), beta.as_weird(), c.as_mut_ptr() as *mut _, c.major_stride()) }
}

/// Hermitian rank-k operation, C = alpha * A * A' + beta * C,
#[inline(always)]
pub unsafe fn herk<A: ?Sized, C: ?Sized>(alpha: <A::Element as Num>::Float, a: &A, beta: <A::Element as Num>::Float, c: &mut C) where A: Matrix, C: Matrix<Element = A::Element>, C::Element: Complex {
    let (cm, cn) = c.dim();
    debug_assert_eq!(cm, cn);

    let n = cm;
    let k = a.dim().1;

    unsafe { A::Element::herk()(a.order() as CBLAS_ORDER, a.uplo() as CBLAS_UPLO, a.transpose() as CBLAS_TRANSPOSE, n, k, alpha, a.as_ptr() as *const _, a.major_stride(), beta, c.as_mut_ptr() as *mut _, c.major_stride()) }
}

/// Hermitian rank-2k operation, C = alpha * A * B' + alpha * B * A' + beta * C
#[inline(always)]
pub unsafe fn her2k<A: ?Sized, B: ?Sized, C: ?Sized>(tran: Transpose, alpha: A::Element, a: &A, b: &B, beta: <A::Element as Num>::Float, c: &mut C) where A: Matrix, B: Matrix<Element = A::Element>, C: Matrix<Element = A::Element>, A::Element: Complex {
    let (cm, cn) = c.dim();
    debug_assert_eq!(cm, cn);

    let n = cn;
    let k = min(a.dim().1, b.dim().1);

    unsafe { A::Element::her2k()(a.order() as CBLAS_ORDER, c.uplo() as CBLAS_UPLO, tran as CBLAS_TRANSPOSE, n, k, alpha.as_weird(), a.as_ptr() as *const _, a.major_stride(), b.as_ptr() as *const _, b.major_stride(), beta, c.as_mut_ptr() as *mut _, c.major_stride()) }
}
