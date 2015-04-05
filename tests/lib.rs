#![feature(core)]

extern crate blas;
extern crate libc;
extern crate num;
extern crate quickcheck;

use std::cmp::{self, Ordering};
use std::fmt;

use blas::{Real, Num};
use quickcheck::quickcheck;

const EPSILON: f64 = 9.765625e-4;

#[derive(PartialEq, PartialOrd)]
struct OrdWr<T>(T);

impl<T: PartialEq> cmp::Eq for OrdWr<T> { }

impl<T: Real> cmp::Ord for OrdWr<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.0 < other.0 {
            Ordering::Less
        } else if self > other {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

fn approx_eq<T: fmt::Debug + Copy + num::Signed + PartialOrd + num::FromPrimitive>(a: T, b: T) -> bool {
    let res = num::abs(a - b) < T::from_f64(EPSILON).unwrap();
    if !res {
        println!("{:?} ~/~ {:?}", a, b);
    }
    res
}

fn dot_prop<T: fmt::Debug + Num + PartialOrd + num::Signed + num::FromPrimitive>(x: Vec<T>, y: Vec<T>) -> bool {
    let expected = x.iter().zip(y.iter()).map(|(&x, &y)| x * y).fold(T::zero(), |acc, item| acc + item);
    let l = cmp::min(x.len(), y.len());
    let actual = blas::dot(&x[..l], &y[..l]);
    approx_eq(actual, expected)
}

#[test]
#[ignore] // FIXME: https://github.com/xianyi/OpenBLAS/issues/522
fn sdot() {
    quickcheck(dot_prop::<f32> as fn(Vec<f32>, Vec<f32>) -> bool);
}

#[test]
fn ddot() {
    quickcheck(dot_prop::<f64> as fn(Vec<f64>, Vec<f64>) -> bool);
}

fn axpy_prop<T: fmt::Debug + Num + PartialOrd + num::Signed + num::FromPrimitive>(a: T, x: Vec<T>, mut y: Vec<T>) -> bool {
    let expected = x.iter().zip(y.iter()).map(|(&x, &y)| y + a * x).collect::<Vec<_>>();
    let l = cmp::min(x.len(), y.len());
    blas::axpy(a, &x[..l], &mut y[..l]);
    expected.into_iter().zip(y.into_iter()).all(|(ex, ac)| approx_eq(ex, ac))
}

#[test]
fn saxpy() {
    quickcheck(axpy_prop::<f32> as fn(f32, Vec<f32>, Vec<f32>) -> bool);
}

#[test]
fn daxpy() {
    quickcheck(axpy_prop::<f64> as fn(f64, Vec<f64>, Vec<f64>) -> bool);
}

fn axpby_prop<T: fmt::Debug + Num + PartialOrd + num::Signed + num::FromPrimitive>(a: T, x: Vec<T>, b: T, mut y: Vec<T>) -> bool {
    let expected = x.iter().zip(y.iter()).map(|(&x, &y)| a * x + b * y).collect::<Vec<_>>();
    let l = cmp::min(x.len(), y.len());
    blas::axpby(a, &x[..l], b, &mut y[..l]);
    expected.into_iter().zip(y.into_iter()).all(|(ex, ac)| approx_eq(ex, ac))
}

#[test]
fn saxpby() {
    quickcheck(axpby_prop::<f32> as fn(f32, Vec<f32>, f32, Vec<f32>) -> bool);
}

#[test]
fn daxpby() {
    quickcheck(axpby_prop::<f64> as fn(f64, Vec<f64>, f64, Vec<f64>) -> bool);
}

fn rot_prop<T: fmt::Debug + Real + PartialOrd + num::Signed + num::FromPrimitive>(mut x: Vec<T>, mut y: Vec<T>, s: T, c: T) -> bool {
    let s = s / T::from_i8(100).unwrap();
    let c = c / T::from_i8(100).unwrap();
    let expected = x.iter().zip(y.iter()).map(|(&x, &y)| (c * x + s * y, -s * x + c * y)).collect::<Vec<_>>();
    let l = cmp::min(x.len(), y.len());
    blas::rot(&mut x[..l], &mut y[..l], c, s);
    expected.into_iter().zip(x.into_iter().zip(y.into_iter())).all(|((exx, exy), (acx, acy))| approx_eq(exx, acx) && approx_eq(exy, acy))
}

#[test]
fn srot() {
    quickcheck(rot_prop::<f32> as fn(Vec<f32>, Vec<f32>, f32, f32) -> bool);
}

#[test]
fn drot() {
    quickcheck(rot_prop::<f64> as fn(Vec<f64>, Vec<f64>, f64, f64) -> bool);
}

// simple smoke test, not very good.
fn rotg_prop<T: fmt::Debug + Real + PartialOrd + num::Signed + num::FromPrimitive>(a: T, b: T) -> bool {
    let (r, _, c, s) = blas::rotg(a, b);
    let mut x = [a]; let mut y = [b];
    blas::rot(&mut x[..], &mut y[..], c, s);
    approx_eq(x[0], r) && approx_eq(y[0], T::from_i8(0).unwrap())
}

#[test]
fn srotg() {
    quickcheck(rotg_prop::<f32> as fn(f32, f32) -> bool);
}

#[test]
fn drotg() {
    quickcheck(rotg_prop::<f64> as fn(f64, f64) -> bool);
}

fn scal_prop<T: fmt::Debug + Real + PartialOrd + num::Signed + num::FromPrimitive>(mut a: Vec<T>, s: T) -> bool {
    let expected = a.iter().map(|&a| a * s).collect::<Vec<_>>();
    blas::scal(s, &mut a[..]);
    expected.into_iter().zip(a.into_iter()).all(|(ex, ac)| approx_eq(ex, ac))
}

#[test]
fn sscal() {
    quickcheck(scal_prop::<f32> as fn(Vec<f32>, f32) -> bool);
}

#[test]
fn dscal() {
    quickcheck(scal_prop::<f64> as fn(Vec<f64>, f64) -> bool);
}

fn asum_prop<T: fmt::Debug + Real + PartialOrd + num::Signed + num::FromPrimitive>(a: Vec<T>) -> bool {
    let expected = a.iter().map(|&a| a.abs()).fold(T::from_i8(0).unwrap(), |acc, it| acc + it);
    let actual = blas::asum(&a[..]);
    approx_eq(expected, actual)
}

#[test]
#[ignore] // FIXME - reliably fails
fn sasum() {
    quickcheck(asum_prop::<f32> as fn(Vec<f32>) -> bool);
}

#[test]
fn dasum() {
    quickcheck(asum_prop::<f64> as fn(Vec<f64>) -> bool);
}

fn iamax_prop<T: fmt::Debug + Real + PartialOrd + num::Signed + num::FromPrimitive>(a: Vec<T>) -> bool {
    let expected = match a.iter().enumerate().max_by(|&(_, elem)| OrdWr(elem.abs())) {
        Some(e) => e.0,
        None => return true
    };
    let actual = blas::iamax(&a[..]);
    expected as libc::size_t == actual
}

#[test]
fn siamax() {
    quickcheck(iamax_prop::<f32> as fn(Vec<f32>) -> bool);
}

#[test]
fn diamax() {
    quickcheck(iamax_prop::<f64> as fn(Vec<f64>) -> bool);
}

fn nrm2_prop<T: fmt::Debug + Real + PartialOrd + num::Signed + num::FromPrimitive>(a: Vec<T>) -> bool {
    let expected = a.iter().fold(T::from_i8(0).unwrap(), |acc, it| acc + it.abs().powi(2)).sqrt();
    let actual = blas::nrm2(&a[..]);
    approx_eq(expected, actual)
}

#[test]
fn snrm2() {
    quickcheck(nrm2_prop::<f32> as fn(Vec<f32>) -> bool);
}

#[test]
fn dnrm2() {
    quickcheck(nrm2_prop::<f64> as fn(Vec<f64>) -> bool);
}
