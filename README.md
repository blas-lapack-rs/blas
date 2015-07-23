# BLAS [![Version][version-img]][version-url] [![Status][status-img]][status-url]

The package provides an interface to the [Basic Linear Algebra Subprograms][1].

## [Documentation][docs]

## Example

```rust
let (m, n, k) = (2, 4, 3);

let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
let b = vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0];
let mut c = vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0];

blas::dgemm(b'N', b'N', m, n, k, 1.0, &a, m, &b, k, 1.0, &mut c, m);

assert_eq!(&c, &vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0]);
```

## Contributing

1. Fork the project.
2. Implement your idea.
3. Open a pull request.

[1]: http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms

[version-img]: http://stainless-steel.github.io/images/crates.svg
[version-url]: https://crates.io/crates/blas
[status-img]: https://travis-ci.org/stainless-steel/blas.svg?branch=master
[status-url]: https://travis-ci.org/stainless-steel/blas
[docs]: https://stainless-steel.github.io/blas
