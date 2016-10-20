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
