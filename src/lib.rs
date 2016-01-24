//! Interface to the [Basic Linear Algebra Subprograms][1].
//!
//! [1]: http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms

extern crate blas_sys as ffi;
extern crate libc;
extern crate num;

#[allow(non_camel_case_types)]
pub type c32 = num::Complex<f32>;

#[allow(non_camel_case_types)]
pub type c64 = num::Complex<f64>;

pub mod c;
pub mod fortran;
