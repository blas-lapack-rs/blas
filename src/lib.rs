//! Interface to the [Basic Linear Algebra Subprograms][blas].
//!
//! ## Configuration
//!
//! The underlying implementation of BLAS to compile, if needed, and link to can
//! be chosen among the following options:
//!
//! * Apple’s [Accelerate framework][accelerate] (macOS only),
//! * Netlib’s [reference implementation][netlib], and
//! * [OpenBLAS][openblas] (default).
//!
//! An implementation can be chosen using the package’s features as follows:
//!
//! ```toml
//! [dependencies]
//! # Apple’s Accelerate framework
//! blas = { version = "0.15", default-features = false, features = ["accelerate"] }
//! # Netlib’s reference implementation
//! blas = { version = "0.15", default-features = false, features = ["netlib"] }
//! # OpenBLAS
//! blas = { version = "0.15", default-features = false, features = ["openblas"] }
//! # OpenBLAS
//! blas = { version = "0.15" }
//! ```
//!
//! [accelerate]: https://developer.apple.com/reference/accelerate
//! [blas]: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
//! [netlib]: http://www.netlib.org/blas
//! [openblas]: http://www.openblas.net

extern crate blas_sys;
extern crate libc;
extern crate num_complex as num;

/// A complex number with 32-bit parts.
#[allow(non_camel_case_types)]
pub type c32 = num::Complex<f32>;

/// A complex number with 64-bit parts.
#[allow(non_camel_case_types)]
pub type c64 = num::Complex<f64>;

pub mod c;
pub mod fortran;
