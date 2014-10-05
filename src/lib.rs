#![feature(macro_rules)]

extern crate libc;

use libc::{c_char, c_double, c_int};

#[link(name = "gfortran")]
#[link(name = "blas", kind = "static")]
extern {
    fn dgemv_(trans: *const c_char, m: *const c_int, n: *const c_int,
        alpha: *const c_double, a: *const c_double, lda: *const c_int,
        x: *const c_double, incx: *const c_int, beta: *const c_double,
        y: *mut c_double, incy: *const c_int);

    fn dgemm_(transa: *const c_char, transb: *const c_char, m: *const c_int,
        n: *const c_int, k: *const c_int, alpha: *const c_double,
        a: *const c_double, lda: *const c_int, b: *const c_double,
        ldb: *const c_int, beta: *const c_double, c: *mut c_double,
        ldc: *const c_int);
}

pub fn dgemv(trans: i8, m: i32, n: i32, alpha: f64, a: &[f64], lda: i32,
    x: &[f64], incx: i32, beta: f64, y: &mut[f64], incy: i32) {

    unsafe {
        dgemv_(&trans, &m, &n, &alpha, a.as_ptr(), &lda, x.as_ptr(), &incx,
            &beta, y.as_mut_ptr(), &incy);
    }
}

pub fn dgemm(transa: i8, transb: i8, m: i32, n: i32, k: i32, alpha: f64,
    a: &[f64], lda: i32, b: &[f64], ldb: i32, beta: f64, c: &mut[f64],
    ldc: i32) {

    unsafe {
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, a.as_ptr(), &lda,
            b.as_ptr(), &ldb, &beta, c.as_mut_ptr(), &ldc);
    }
}

#[cfg(test)]
mod tests {
    extern crate test;

    use self::test::Bencher;

    macro_rules! assert_equal(
        ($given:expr , $expected:expr) => ({
            assert_eq!($given.len(), $expected.len());
            for i in range(0u, $given.len()) {
                assert_eq!($given[i], $expected[i]);
            }
        });
    )

    #[test]
    fn dgemv() {
        let m: i32 = 2;
        let n: i32 = 3;

        let a = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let x = [1.0, 2.0, 3.0];
        let mut y = [6.0, 8.0];

        super::dgemv('N' as i8, m, n, 1.0, a, m, x, 1, 1.0, y, 1);

        let expected_y = [20.0, 40.0];

        assert_equal!(y, expected_y);
    }

    #[test]
    fn dgemm() {
        let m: i32 = 2;
        let n: i32 = 4;
        let k: i32 = 3;

        let a = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let b = [1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0];
        let mut c = [2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0];

        super::dgemm('N' as i8, 'N' as i8, m, n, k, 1.0, a, m, b, k, 1.0, c, m);

        let expected_c = [40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0];

        assert_equal!(c, expected_c);
    }

    #[bench]
    fn dgemv_few_large(b: &mut Bencher) {
        #[allow(non_uppercase_statics)]
        static m: uint = 1000;

        let a = box [0.0, ..m*m];
        let x = box [0.0, ..m*1];
        let mut y = box [0.0, ..m*1];

        b.iter(|| {
            super::dgemv('N' as i8, m as i32, m as i32, 1.0, &*a,
                m as i32, &*x, 1, 1.0, &mut *y, 1)
        })
    }

    #[bench]
    fn dgemv_many_small(b: &mut Bencher) {
        #[allow(non_uppercase_statics)]
        static m: uint = 20;

        let a = box [0.0, ..m*m];
        let x = box [0.0, ..m*1];
        let mut y = box [0.0, ..m*1];

        b.iter(|| {
            for _ in range(0u, 20000) {
                super::dgemv('N' as i8, m as i32, m as i32, 1.0, &*a,
                    m as i32, &*x, 1, 1.0, &mut *y, 1);
            }
        })
    }
}
