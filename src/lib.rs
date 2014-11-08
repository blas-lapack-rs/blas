//! An interface to the [Basic Linear Algebra Subprograms][1].
//!
//! [1]: http://www.netlib.org/blas/

#![feature(phase)]
#![allow(non_snake_case)]

extern crate "libblas-sys" as raw;

/// http://www.netlib.org/lapack/explore-html/dc/da8/dgemv_8f.html
#[inline]
pub fn dgemv(TRANS: u8, M: uint, N: uint, ALPHA: f64, A: &[f64], LDA: uint,
             X: &[f64], INCX: uint, BETA: f64, Y: &mut [f64], INCY: uint) {

    unsafe {
        raw::dgemv(&(TRANS as i8), &(M as i32), &(N as i32), &ALPHA,
                   A.as_ptr(), &(LDA as i32), X.as_ptr(), &(INCX as i32),
                   &BETA, Y.as_mut_ptr(), &(INCY as i32));
    }
}

/// http://www.netlib.org/lapack/explore-html/d7/d2b/dgemm_8f.html
#[inline]
pub fn dgemm(TRANSA: u8, TRANSB: u8, M: uint, N: uint, K: uint, ALPHA: f64, A: &[f64],
             LDA: uint, B: &[f64], LDB: uint, BETA: f64, C: &mut [f64], LDC: uint) {

    unsafe {
        raw::dgemm(&(TRANSA as i8), &(TRANSB as i8), &(M as i32), &(N as i32),
                   &(K as i32), &ALPHA, A.as_ptr(), &(LDA as i32), B.as_ptr(),
                   &(LDB as i32), &BETA, C.as_mut_ptr(), &(LDC as i32));
    }
}

#[cfg(test)]
mod test {
    #[phase(plugin)] extern crate assert;

    #[test]
    fn dgemv() {
        let (M, N) = (2, 3);

        let A = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let X = vec![1.0, 2.0, 3.0];
        let mut Y = vec![6.0, 8.0];

        ::dgemv(b'N', M, N, 1.0, A.as_slice(), M, X.as_slice(), 1, 1.0,
                Y.as_mut_slice(), 1);

        let expected_Y = vec![20.0, 40.0];
        assert_equal!(Y, expected_Y);
    }

    #[test]
    fn dgemm() {
        let (M, N, K) = (2, 4, 3);

        let A = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let B = vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0];
        let mut C = vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0];

        ::dgemm(b'N', b'N', M, N, K, 1.0, A.as_slice(), M, B.as_slice(), K, 1.0,
                C.as_mut_slice(), M);

        let expected_C = vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0];
        assert_equal!(C, expected_C);
    }
}

#[cfg(test)]
mod bench {
    extern crate test;

    #[bench]
    fn dgemv_few_large(bench: &mut test::Bencher) {
        let M = 1000;

        let A = Vec::from_elem(M * M, 1.0);
        let X = Vec::from_elem(M * 1, 1.0);
        let mut Y = Vec::from_elem(M * 1, 1.0);

        bench.iter(|| {
            ::dgemv(b'N', M, M, 1.0, A.as_slice(), M, X.as_slice(), 1, 1.0,
                    Y.as_mut_slice(), 1)
        });
    }

    #[bench]
    fn dgemv_many_small(bench: &mut test::Bencher) {
        let M = 20;

        let A = Vec::from_elem(M * M, 1.0);
        let X = Vec::from_elem(M * 1, 1.0);
        let mut Y = Vec::from_elem(M * 1, 1.0);

        bench.iter(|| {
            for _ in range(0u, 20000) {
                ::dgemv(b'N', M, M, 1.0, A.as_slice(), M, X.as_slice(), 1, 1.0,
                        Y.as_mut_slice(), 1);
            }
        });
    }
}
