#![feature(macro_rules)]

extern crate blas;

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
    let (m, n) = (2, 3);

    let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let x = vec![1.0, 2.0, 3.0];
    let mut y = vec![6.0, 8.0];

    blas::dgemv(blas::NORMAL, m, n, 1.0, a.as_ptr(), m, x.as_ptr(), 1,
        1.0, y.as_mut_ptr(), 1);

    let expected_y = vec![20.0, 40.0];
    assert_equal!(y, expected_y);
}

#[test]
fn dgemm() {
    let (m, n, k) = (2, 4, 3);

    let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let b = vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0];
    let mut c = vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0];

    blas::dgemm(blas::NORMAL, blas::NORMAL, m, n, k, 1.0, a.as_ptr(),
        m, b.as_ptr(), k, 1.0, c.as_mut_ptr(), m);

    let expected_c = vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0];
    assert_equal!(c, expected_c);
}
