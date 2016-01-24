use ffi;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub enum Diagonal {
    NonUnit = 131,
    Unit = 132,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub enum Major {
    Row = 101,
    Column = 102,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub enum Triangular {
    Upper = 121,
    Lower = 122,
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
    Diagonal => CblasDiag,
    Major => CblasLayout,
    Side => CblasSide,
    Transpose => CblasTranspose,
    Triangular => CblasUplo,
}
