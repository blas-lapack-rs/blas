//! Interface to the [Basic Linear Algebra Subprograms][1].
//!
//! ## Example
//!
//! ```
//! let (m, n, k) = (2, 4, 3);
//!
//! let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
//! let b = vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0];
//! let mut c = vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0];
//!
//! blas::dgemm(b'N', b'N', m, n, k, 1.0, &a, m, &b, k, 1.0, &mut c, m);
//!
//! assert_eq!(&c, &vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0]);
//! ```
//!
//! [1]: http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms

extern crate blas_sys as ffi;
extern crate libc;
extern crate num;

#[allow(non_camel_case_types)]
pub type c32 = num::Complex<f32>;

#[allow(non_camel_case_types)]
pub type c64 = num::Complex<f64>;

mod fortran;

pub use fortran::*;
