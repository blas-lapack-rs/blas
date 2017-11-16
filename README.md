# BLAS [![Package][package-img]][package-url] [![Documentation][documentation-img]][documentation-url] [![Build][build-img]][build-url]

The package provides wrappers for [BLAS] (Fortran).

## Example (C)

```rust
use blas::c::*;

let (m, n, k) = (2, 4, 3);
let a = vec![
    1.0, 4.0,
    2.0, 5.0,
    3.0, 6.0,
];
let b = vec![
    1.0, 5.0,  9.0,
    2.0, 6.0, 10.0,
    3.0, 7.0, 11.0,
    4.0, 8.0, 12.0,
];
let mut c = vec![
    2.0, 7.0,
    6.0, 2.0,
    0.0, 7.0,
    4.0, 2.0,
];

unsafe {
    dgemm(
        Layout::ColumnMajor,
        Transpose::None,
        Transpose::None,
        m,
        n,
        k,
        1.0,
        &a,
        m,
        &b,
        k,
        1.0,
        &mut c,
        m,
    );
}

assert_eq!(
    &c,
    &vec![
        40.0,  90.0,
        50.0, 100.0,
        50.0, 120.0,
        60.0, 130.0,
    ]
);
```

## Example (Fortran)

```rust
use blas::fortran::*;

let (m, n, k) = (2, 4, 3);
let a = vec![
    1.0, 4.0,
    2.0, 5.0,
    3.0, 6.0,
];
let b = vec![
    1.0, 5.0,  9.0,
    2.0, 6.0, 10.0,
    3.0, 7.0, 11.0,
    4.0, 8.0, 12.0,
];
let mut c = vec![
    2.0, 7.0,
    6.0, 2.0,
    0.0, 7.0,
    4.0, 2.0,
];

unsafe {
    dgemm(b'N', b'N', m, n, k, 1.0, &a, m, &b, k, 1.0, &mut c, m);
}

assert_eq!(
    &c,
    &vec![
        40.0,  90.0,
        50.0, 100.0,
        50.0, 120.0,
        60.0, 130.0,
    ]
);
```

## Contribution

Your contribution is highly appreciated. Do not hesitate to open an issue or a
pull request. Note that any contribution submitted for inclusion in the project
will be licensed according to the terms given in [LICENSE.md](LICENSE.md).

[blas]: https://en.wikipedia.org/wiki/BLAS

[build-img]: https://travis-ci.org/blas-lapack-rs/blas.svg?branch=master
[build-url]: https://travis-ci.org/blas-lapack-rs/blas
[documentation-img]: https://docs.rs/blas/badge.svg
[documentation-url]: https://docs.rs/blas
[package-img]: https://img.shields.io/crates/v/blas.svg
[package-url]: https://crates.io/crates/blas
