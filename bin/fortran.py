#!/usr/bin/env python

from function import Function
import re

level_scalars = {
    1: ["alpha", "a", "b", "c", "s", "d1", "d2", "x1", "y1"],
    2: ["alpha", "beta"],
    3: ["alpha", "beta"],
}

level1 = """
    // Single
    pub fn srotg_(a: *mut c_float, b: *mut c_float, c: *mut c_float, s: *mut c_float);
    pub fn srotmg_(d1: *mut c_float, d2: *mut c_float, x1: *mut c_float, y1: *const c_float,
                   param: *mut c_float);
    pub fn srot_(n: *const c_int, x: *mut c_float, incx: *const c_int, y: *mut c_float,
                 incy: *const c_int, c: *const c_float, s: *const c_float);
    pub fn srotm_(n: *const c_int, x: *mut c_float, incx: *const c_int, y: *mut c_float,
                  incy: *const c_int, param: *const c_float);
    pub fn sswap_(n: *const c_int, x: *mut c_float, incx: *const c_int, y: *mut c_float,
                  incy: *const c_int);
    pub fn sscal_(n: *const c_int, a: *const c_float, x: *mut c_float, incx: *const c_int);
    pub fn scopy_(n: *const c_int, x: *const c_float, incx: *const c_int, y: *mut c_float,
                  incy: *const c_int);
    pub fn saxpy_(n: *const c_int, alpha: *const c_float, x: *const c_float, incx: *const c_int,
                  y: *mut c_float, incy: *const c_int);
    pub fn sdot_(n: *const c_int, x: *const c_float, incx: *const c_int, y: *const c_float,
                 incy: *const c_int) -> c_float;
    pub fn sdsdot_(n: *const c_int, sb: *const c_float, x: *const c_float, incx: *const c_int,
                   y: *const c_float, incy: *const c_int) -> c_float;
    pub fn snrm2_(n: *const c_int, x: *const c_float, incx: *const c_int) -> c_float;
    pub fn scnrm2_(n: *const c_int, x: *const c_float_complex, incx: *const c_int) -> c_float;
    pub fn sasum_(n: *const c_int, x: *const c_float, incx: *const c_int) -> c_float;
    pub fn isamax_(n: *const c_int, x: *const c_float, incx: *const c_int) -> c_int;

    // Double
    pub fn drotg_(a: *mut c_double, b: *mut c_double, c: *mut c_double, s: *mut c_double);
    pub fn drotmg_(d1: *mut c_double, d2: *mut c_double, x1: *mut c_double, y1: *const c_double,
                   param: *mut c_double);
    pub fn drot_(n: *const c_int, x: *mut c_double, incx: *const c_int, y: *mut c_double,
                 incy: *const c_int, c: *const c_double, s: *const c_double);
    pub fn drotm_(n: *const c_int, x: *mut c_double, incx: *const c_int, y: *mut c_double,
                  incy: *const c_int, param: *const c_double);
    pub fn dswap_(n: *const c_int, x: *mut c_double, incx: *const c_int, y: *mut c_double,
                  incy: *const c_int);
    pub fn dscal_(n: *const c_int, a: *const c_double, x: *mut c_double, incx: *const c_int);
    pub fn dcopy_(n: *const c_int, x: *const c_double, incx: *const c_int, y: *mut c_double,
                  incy: *const c_int);
    pub fn daxpy_(n: *const c_int, alpha: *const c_double, x: *const c_double, incx: *const c_int,
                  y: *mut c_double, incy: *const c_int);
    pub fn ddot_(n: *const c_int, x: *const c_double, incx: *const c_int, y: *const c_double,
                 incy: *const c_int) -> c_double;
    pub fn dsdot_(n: *const c_int, x: *const c_float, incx: *const c_int, y: *const c_float,
                  incy: *const c_int) -> c_double;
    pub fn dnrm2_(n: *const c_int, x: *const c_double, incx: *const c_int) -> c_double;
    pub fn dznrm2_(n: *const c_int, x: *const c_double_complex, incx: *const c_int) -> c_double;
    pub fn dasum_(n: *const c_int, x: *const c_double, incx: *const c_int) -> c_double;
    pub fn idamax_(n: *const c_int, x: *const c_double, incx: *const c_int) -> c_int;

    // Complex
    pub fn crotg_(a: *mut c_float_complex, b: *const c_float_complex, c: *mut c_float,
                  s: *mut c_float_complex);
    pub fn csrot_(n: *const c_int, x: *mut c_float_complex, incx: *const c_int,
                  y: *mut c_float_complex, incy: *const c_int, c: *const c_float,
                  s: *const c_float);
    pub fn cswap_(n: *const c_int, x: *mut c_float_complex, incx: *const c_int,
                  y: *mut c_float_complex, incy: *const c_int);
    pub fn cscal_(n: *const c_int, a: *const c_float_complex, x: *mut c_float_complex,
                  incx: *const c_int);
    pub fn csscal_(n: *const c_int, a: *const c_float, x: *mut c_float_complex,
                   incx: *const c_int);
    pub fn ccopy_(n: *const c_int, x: *const c_float_complex, incx: *const c_int,
                  y: *mut c_float_complex, incy: *const c_int);
    pub fn caxpy_(n: *const c_int, alpha: *const c_float_complex, x: *const c_float_complex,
                  incx: *const c_int, y: *mut c_float_complex, incy: *const c_int);
    pub fn cdotu_(pres: *mut c_float_complex, n: *const c_int, x: *const c_float_complex,
                  incx: *const c_int, y: *const c_float_complex, incy: *const c_int);
    pub fn cdotc_(pres: *mut c_float_complex, n: *const c_int, x: *const c_float_complex,
                  incx: *const c_int, y: *const c_float_complex, incy: *const c_int);
    pub fn scasum_(n: *const c_int, x: *const c_float_complex, incx: *const c_int) -> c_float;
    pub fn icamax_(n: *const c_int, x: *const c_float_complex, incx: *const c_int) -> c_int;

    // Double complex
    pub fn zrotg_(a: *mut c_double_complex, b: *const c_double_complex, c: *mut c_double,
                  s: *mut c_double_complex);
    pub fn zdrot_(n: *const c_int, x: *mut c_double_complex, incx: *const c_int,
                  y: *mut c_double_complex, incy: *const c_int, c: *const c_double,
                  s: *const c_double);
    pub fn zswap_(n: *const c_int, x: *mut c_double_complex, incx: *const c_int,
                  y: *mut c_double_complex, incy: *const c_int);
    pub fn zscal_(n: *const c_int, a: *const c_double_complex, x: *mut c_double_complex,
                  incx: *const c_int);
    pub fn zdscal_(n: *const c_int, a: *const c_double, x: *mut c_double_complex,
                   incx: *const c_int);
    pub fn zcopy_(n: *const c_int, x: *const c_double_complex, incx: *const c_int,
                  y: *mut c_double_complex, incy: *const c_int);
    pub fn zaxpy_(n: *const c_int, alpha: *const c_double_complex, x: *const c_double_complex,
                  incx: *const c_int, y: *mut c_double_complex, incy: *const c_int);
    pub fn zdotu_(pres: *mut c_double_complex, n: *const c_int, x: *const c_double_complex,
                  incx: *const c_int, y: *const c_double_complex, incy: *const c_int);
    pub fn zdotc_(pres: *mut c_double_complex, n: *const c_int, x: *const c_double_complex,
                  incx: *const c_int, y: *const c_double_complex, incy: *const c_int);
    pub fn dzasum_(n: *const c_int, x: *const c_double_complex, incx: *const c_int) -> c_double;
    pub fn izamax_(n: *const c_int, x: *const c_double_complex, incx: *const c_int) -> c_int;
"""

level2 = """
    // Single
    pub fn sgemv_(trans: *const c_char, m: *const c_int, n: *const c_int, alpha: *const c_float,
                  a: *const c_float, lda: *const c_int, x: *const c_float, incx: *const c_int,
                  beta: *const c_float, y: *mut c_float, incy: *const c_int);
    pub fn sgbmv_(trans: *const c_char, m: *const c_int, n: *const c_int, kl: *const c_int,
                  ku: *const c_int, alpha: *const c_float, a: *const c_float, lda: *const c_int,
                  x: *const c_float, incx: *const c_int, beta: *const c_float, y: *mut c_float,
                  incy: *const c_int);
    pub fn ssymv_(uplo: *const c_char, n: *const c_int, alpha: *const c_float, a: *const c_float,
                  lda: *const c_int, x: *const c_float, incx: *const c_int, beta: *const c_float,
                  y: *mut c_float, incy: *const c_int);
    pub fn ssbmv_(uplo: *const c_char, n: *const c_int, k: *const c_int, alpha: *const c_float,
                  a: *const c_float, lda: *const c_int, x: *const c_float, incx: *const c_int,
                  beta: *const c_float, y: *mut c_float, incy: *const c_int);
    pub fn sspmv_(uplo: *const c_char, n: *const c_int, alpha: *const c_float, ap: *const c_float,
                  x: *const c_float, incx: *const c_int, beta: *const c_float, y: *mut c_float,
                  incy: *const c_int);
    pub fn strmv_(uplo: *const c_char, transa: *const c_char, diag: *const c_char, n: *const c_int,
                  a: *const c_float, lda: *const c_int, b: *mut c_float, incx: *const c_int);
    pub fn stbmv_(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int,
                  k: *const c_int, a: *const c_float, lda: *const c_int, x: *mut c_float,
                  incx: *const c_int);
    pub fn stpmv_(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int,
                  ap: *const c_float, x: *mut c_float, incx: *const c_int);
    pub fn strsv_(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int,
                  a: *const c_float, lda: *const c_int, x: *mut c_float, incx: *const c_int);
    pub fn stbsv_(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int,
                  k: *const c_int, a: *const c_float, lda: *const c_int, x: *mut c_float,
                  incx: *const c_int);
    pub fn stpsv_(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int,
                  ap: *const c_float, x: *mut c_float, incx: *const c_int);
    pub fn sger_(m: *const c_int, n: *const c_int, alpha: *const c_float, x: *const c_float,
                 incx: *const c_int, y: *const c_float, incy: *const c_int, a: *mut c_float,
                 lda: *const c_int);
    pub fn ssyr_(uplo: *const c_char, n: *const c_int, alpha: *const c_float, x: *const c_float,
                 incx: *const c_int, a: *mut c_float, lda: *const c_int);
    pub fn sspr_(uplo: *const c_char, n: *const c_int, alpha: *const c_float, x: *const c_float,
                 incx: *const c_int, ap: *mut c_float);
    pub fn ssyr2_(uplo: *const c_char, n: *const c_int, alpha: *const c_float, x: *const c_float,
                  incx: *const c_int, y: *const c_float, incy: *const c_int, a: *mut c_float,
                  lda: *const c_int);
    pub fn sspr2_(uplo: *const c_char, n: *const c_int, alpha: *const c_float, x: *const c_float,
                  incx: *const c_int, y: *const c_float, incy: *const c_int, ap: *mut c_float);

    // Double
    pub fn dgemv_(trans: *const c_char, m: *const c_int, n: *const c_int, alpha: *const c_double,
                  a: *const c_double, lda: *const c_int, x: *const c_double, incx: *const c_int,
                  beta: *const c_double, y: *mut c_double, incy: *const c_int);
    pub fn dgbmv_(trans: *const c_char, m: *const c_int, n: *const c_int, kl: *const c_int,
                  ku: *const c_int, alpha: *const c_double, a: *const c_double, lda: *const c_int,
                  x: *const c_double, incx: *const c_int, beta: *const c_double, y: *mut c_double,
                  incy: *const c_int);
    pub fn dsymv_(uplo: *const c_char, n: *const c_int, alpha: *const c_double, a: *const c_double,
                  lda: *const c_int, x: *const c_double, incx: *const c_int, beta: *const c_double,
                  y: *mut c_double, incy: *const c_int);
    pub fn dsbmv_(uplo: *const c_char, n: *const c_int, k: *const c_int, alpha: *const c_double,
                  a: *const c_double, lda: *const c_int, x: *const c_double, incx: *const c_int,
                  beta: *const c_double, y: *mut c_double, incy: *const c_int);
    pub fn dspmv_(uplo: *const c_char, n: *const c_int, alpha: *const c_double,
                  ap: *const c_double, x: *const c_double, incx: *const c_int,
                  beta: *const c_double, y: *mut c_double, incy: *const c_int);
    pub fn dtrmv_(uplo: *const c_char, transa: *const c_char, diag: *const c_char, n: *const c_int,
                  a: *const c_double, lda: *const c_int, b: *mut c_double, incx: *const c_int);
    pub fn dtbmv_(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int,
                  k: *const c_int, a: *const c_double, lda: *const c_int, x: *mut c_double,
                  incx: *const c_int);
    pub fn dtpmv_(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int,
                  ap: *const c_double, x: *mut c_double, incx: *const c_int);
    pub fn dtrsv_(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int,
                  a: *const c_double, lda: *const c_int, x: *mut c_double, incx: *const c_int);
    pub fn dtbsv_(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int,
                  k: *const c_int, a: *const c_double, lda: *const c_int, x: *mut c_double,
                  incx: *const c_int);
    pub fn dtpsv_(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int,
                  ap: *const c_double, x: *mut c_double, incx: *const c_int);
    pub fn dger_(m: *const c_int, n: *const c_int, alpha: *const c_double, x: *const c_double,
                 incx: *const c_int, y: *const c_double, incy: *const c_int, a: *mut c_double,
                 lda: *const c_int);
    pub fn dsyr_(uplo: *const c_char, n: *const c_int, alpha: *const c_double, x: *const c_double,
                 incx: *const c_int, a: *mut c_double, lda: *const c_int);
    pub fn dspr_(uplo: *const c_char, n: *const c_int, alpha: *const c_double, x: *const c_double,
                 incx: *const c_int, ap: *mut c_double);
    pub fn dsyr2_(uplo: *const c_char, n: *const c_int, alpha: *const c_double, x: *const c_double,
                  incx: *const c_int, y: *const c_double, incy: *const c_int, a: *mut c_double,
                  lda: *const c_int);
    pub fn dspr2_(uplo: *const c_char, n: *const c_int, alpha: *const c_double, x: *const c_double,
                  incx: *const c_int, y: *const c_double, incy: *const c_int, ap: *mut c_double);

    // Complex
    pub fn cgemv_(trans: *const c_char, m: *const c_int, n: *const c_int,
                  alpha: *const c_float_complex, a: *const c_float_complex, lda: *const c_int,
                  x: *const c_float_complex, incx: *const c_int, beta: *const c_float_complex,
                  y: *mut c_float_complex, incy: *const c_int);
    pub fn cgbmv_(trans: *const c_char, m: *const c_int, n: *const c_int, kl: *const c_int,
                  ku: *const c_int, alpha: *const c_float_complex, a: *const c_float_complex,
                  lda: *const c_int, x: *const c_float_complex, incx: *const c_int,
                  beta: *const c_float_complex, y: *mut c_float_complex, incy: *const c_int);
    pub fn chemv_(uplo: *const c_char, n: *const c_int, alpha: *const c_float_complex,
                  a: *const c_float_complex, lda: *const c_int, x: *const c_float_complex,
                  incx: *const c_int, beta: *const c_float_complex, y: *mut c_float_complex,
                  incy: *const c_int);
    pub fn chbmv_(uplo: *const c_char, n: *const c_int, k: *const c_int,
                  alpha: *const c_float_complex, a: *const c_float_complex, lda: *const c_int,
                  x: *const c_float_complex, incx: *const c_int, beta: *const c_float_complex,
                  y: *mut c_float_complex, incy: *const c_int);
    pub fn chpmv_(uplo: *const c_char, n: *const c_int, alpha: *const c_float_complex,
                  ap: *const c_float_complex, x: *const c_float_complex, incx: *const c_int,
                  beta: *const c_float_complex, y: *mut c_float_complex, incy: *const c_int);
    pub fn ctrmv_(uplo: *const c_char, transa: *const c_char, diag: *const c_char, n: *const c_int,
                  a: *const c_float_complex, lda: *const c_int, b: *mut c_float_complex,
                  incx: *const c_int);
    pub fn ctbmv_(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int,
                  k: *const c_int, a: *const c_float_complex, lda: *const c_int,
                  x: *mut c_float_complex, incx: *const c_int);
    pub fn ctpmv_(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int,
                  ap: *const c_float_complex, x: *mut c_float_complex, incx: *const c_int);
    pub fn ctrsv_(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int,
                  a: *const c_float_complex, lda: *const c_int, x: *mut c_float_complex,
                  incx: *const c_int);
    pub fn ctbsv_(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int,
                  k: *const c_int, a: *const c_float_complex, lda: *const c_int,
                  x: *mut c_float_complex, incx: *const c_int);
    pub fn ctpsv_(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int,
                  ap: *const c_float_complex, x: *mut c_float_complex, incx: *const c_int);
    pub fn cgeru_(m: *const c_int, n: *const c_int, alpha: *const c_float_complex,
                  x: *const c_float_complex, incx: *const c_int, y: *const c_float_complex,
                  incy: *const c_int, a: *mut c_float_complex, lda: *const c_int);
    pub fn cgerc_(m: *const c_int, n: *const c_int, alpha: *const c_float_complex,
                  x: *const c_float_complex, incx: *const c_int, y: *const c_float_complex,
                  incy: *const c_int, a: *mut c_float_complex, lda: *const c_int);
    pub fn cher_(uplo: *const c_char, n: *const c_int, alpha: *const c_float,
                 x: *const c_float_complex, incx: *const c_int, a: *mut c_float_complex,
                 lda: *const c_int);
    pub fn chpr_(uplo: *const c_char, n: *const c_int, alpha: *const c_float,
                 x: *const c_float_complex, incx: *const c_int, ap: *mut c_float_complex);
    pub fn chpr2_(uplo: *const c_char, n: *const c_int, alpha: *const c_float_complex,
                  x: *const c_float_complex, incx: *const c_int, y: *const c_float_complex,
                  incy: *const c_int, ap: *mut c_float_complex);
    pub fn cher2_(uplo: *const c_char, n: *const c_int, alpha: *const c_float_complex,
                  x: *const c_float_complex, incx: *const c_int, y: *const c_float_complex,
                  incy: *const c_int, a: *mut c_float_complex, lda: *const c_int);

    // Double complex
    pub fn zgemv_(trans: *const c_char, m: *const c_int, n: *const c_int,
                  alpha: *const c_double_complex, a: *const c_double_complex, lda: *const c_int,
                  x: *const c_double_complex, incx: *const c_int, beta: *const c_double_complex,
                  y: *mut c_double_complex, incy: *const c_int);
    pub fn zgbmv_(trans: *const c_char, m: *const c_int, n: *const c_int, kl: *const c_int,
                  ku: *const c_int, alpha: *const c_double_complex, a: *const c_double_complex,
                  lda: *const c_int, x: *const c_double_complex, incx: *const c_int,
                  beta: *const c_double_complex, y: *mut c_double_complex, incy: *const c_int);
    pub fn zhemv_(uplo: *const c_char, n: *const c_int, alpha: *const c_double_complex,
                  a: *const c_double_complex, lda: *const c_int, x: *const c_double_complex,
                  incx: *const c_int, beta: *const c_double_complex, y: *mut c_double_complex,
                  incy: *const c_int);
    pub fn zhbmv_(uplo: *const c_char, n: *const c_int, k: *const c_int,
                  alpha: *const c_double_complex, a: *const c_double_complex, lda: *const c_int,
                  x: *const c_double_complex, incx: *const c_int, beta: *const c_double_complex,
                  y: *mut c_double_complex, incy: *const c_int);
    pub fn zhpmv_(uplo: *const c_char, n: *const c_int, alpha: *const c_double_complex,
                  ap: *const c_double_complex, x: *const c_double_complex, incx: *const c_int,
                  beta: *const c_double_complex, y: *mut c_double_complex, incy: *const c_int);
    pub fn ztrmv_(uplo: *const c_char, transa: *const c_char, diag: *const c_char, n: *const c_int,
                  a: *const c_double_complex, lda: *const c_int, b: *mut c_double_complex,
                  incx: *const c_int);
    pub fn ztbmv_(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int,
                  k: *const c_int, a: *const c_double_complex, lda: *const c_int,
                  x: *mut c_double_complex, incx: *const c_int);
    pub fn ztpmv_(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int,
                  ap: *const c_double_complex, x: *mut c_double_complex, incx: *const c_int);
    pub fn ztrsv_(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int,
                  a: *const c_double_complex, lda: *const c_int, x: *mut c_double_complex,
                  incx: *const c_int);
    pub fn ztbsv_(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int,
                  k: *const c_int, a: *const c_double_complex, lda: *const c_int,
                  x: *mut c_double_complex, incx: *const c_int);
    pub fn ztpsv_(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int,
                  ap: *const c_double_complex, x: *mut c_double_complex, incx: *const c_int);
    pub fn zgeru_(m: *const c_int, n: *const c_int, alpha: *const c_double_complex,
                  x: *const c_double_complex, incx: *const c_int, y: *const c_double_complex,
                  incy: *const c_int, a: *mut c_double_complex, lda: *const c_int);
    pub fn zgerc_(m: *const c_int, n: *const c_int, alpha: *const c_double_complex,
                  x: *const c_double_complex, incx: *const c_int, y: *const c_double_complex,
                  incy: *const c_int, a: *mut c_double_complex, lda: *const c_int);
    pub fn zher_(uplo: *const c_char, n: *const c_int, alpha: *const c_double,
                 x: *const c_double_complex, incx: *const c_int, a: *mut c_double_complex,
                 lda: *const c_int);
    pub fn zhpr_(uplo: *const c_char, n: *const c_int, alpha: *const c_double,
                 x: *const c_double_complex, incx: *const c_int, ap: *mut c_double_complex);
    pub fn zher2_(uplo: *const c_char, n: *const c_int, alpha: *const c_double_complex,
                  x: *const c_double_complex, incx: *const c_int, y: *const c_double_complex,
                  incy: *const c_int, a: *mut c_double_complex, lda: *const c_int);
    pub fn zhpr2_(uplo: *const c_char, n: *const c_int, alpha: *const c_double_complex,
                  x: *const c_double_complex, incx: *const c_int, y: *const c_double_complex,
                  incy: *const c_int, ap: *mut c_double_complex);
"""

level3 = """
    // Single
    pub fn sgemm_(transa: *const c_char, transb: *const c_char, m: *const c_int, n: *const c_int,
                  k: *const c_int, alpha: *const c_float, a: *const c_float, lda: *const c_int,
                  b: *const c_float, ldb: *const c_int, beta: *const c_float, c: *mut c_float,
                  ldc: *const c_int);
    pub fn ssymm_(side: *const c_char, uplo: *const c_char, m: *const c_int, n: *const c_int,
                  alpha: *const c_float, a: *const c_float, lda: *const c_int, b: *const c_float,
                  ldb: *const c_int, beta: *const c_float, c: *mut c_float, ldc: *const c_int);
    pub fn ssyrk_(uplo: *const c_char, trans: *const c_char, n: *const c_int, k: *const c_int,
                  alpha: *const c_float, a: *const c_float, lda: *const c_int,
                  beta: *const c_float, c: *mut c_float, ldc: *const c_int);
    pub fn ssyr2k_(uplo: *const c_char, trans: *const c_char, n: *const c_int, k: *const c_int,
                   alpha: *const c_float, a: *const c_float, lda: *const c_int, b: *const c_float,
                   ldb: *const c_int, beta: *const c_float, c: *mut c_float, ldc: *const c_int);
    pub fn strmm_(side: *const c_char, uplo: *const c_char, transa: *const c_char,
                  diag: *const c_char, m: *const c_int, n: *const c_int, alpha: *const c_float,
                  a: *const c_float, lda: *const c_int, b: *mut c_float, ldb: *const c_int);
    pub fn strsm_(side: *const c_char, uplo: *const c_char, transa: *const c_char,
                  diag: *const c_char, m: *const c_int, n: *const c_int, alpha: *const c_float,
                  a: *const c_float, lda: *const c_int, b: *mut c_float, ldb: *const c_int);

    // Double
    pub fn dgemm_(transa: *const c_char, transb: *const c_char, m: *const c_int, n: *const c_int,
                  k: *const c_int, alpha: *const c_double, a: *const c_double, lda: *const c_int,
                  b: *const c_double, ldb: *const c_int, beta: *const c_double, c: *mut c_double,
                  ldc: *const c_int);
    pub fn dsymm_(side: *const c_char, uplo: *const c_char, m: *const c_int, n: *const c_int,
                  alpha: *const c_double, a: *const c_double, lda: *const c_int,
                  b: *const c_double, ldb: *const c_int, beta: *const c_double, c: *mut c_double,
                  ldc: *const c_int);
    pub fn dsyrk_(uplo: *const c_char, trans: *const c_char, n: *const c_int, k: *const c_int,
                  alpha: *const c_double, a: *const c_double, lda: *const c_int,
                  beta: *const c_double, c: *mut c_double, ldc: *const c_int);
    pub fn dsyr2k_(uplo: *const c_char, trans: *const c_char, n: *const c_int, k: *const c_int,
                   alpha: *const c_double, a: *const c_double, lda: *const c_int,
                   b: *const c_double, ldb: *const c_int, beta: *const c_double, c: *mut c_double,
                   ldc: *const c_int);
    pub fn dtrmm_(side: *const c_char, uplo: *const c_char, transa: *const c_char,
                  diag: *const c_char, m: *const c_int, n: *const c_int, alpha: *const c_double,
                  a: *const c_double, lda: *const c_int, b: *mut c_double, ldb: *const c_int);
    pub fn dtrsm_(side: *const c_char, uplo: *const c_char, transa: *const c_char,
                  diag: *const c_char, m: *const c_int, n: *const c_int, alpha: *const c_double,
                  a: *const c_double, lda: *const c_int, b: *mut c_double, ldb: *const c_int);

    // Complex
    pub fn cgemm_(transa: *const c_char, transb: *const c_char, m: *const c_int, n: *const c_int,
                  k: *const c_int, alpha: *const c_float_complex, a: *const c_float_complex,
                  lda: *const c_int, b: *const c_float_complex, ldb: *const c_int,
                  beta: *const c_float_complex, c: *mut c_float_complex, ldc: *const c_int);
    pub fn csymm_(side: *const c_char, uplo: *const c_char, m: *const c_int, n: *const c_int,
                  alpha: *const c_float_complex, a: *const c_float_complex, lda: *const c_int,
                  b: *const c_float_complex, ldb: *const c_int, beta: *const c_float_complex,
                  c: *mut c_float_complex, ldc: *const c_int);
    pub fn chemm_(side: *const c_char, uplo: *const c_char, m: *const c_int, n: *const c_int,
                  alpha: *const c_float_complex, a: *const c_float_complex, lda: *const c_int,
                  b: *const c_float_complex, ldb: *const c_int, beta: *const c_float_complex,
                  c: *mut c_float_complex, ldc: *const c_int);
    pub fn csyrk_(uplo: *const c_char, trans: *const c_char, n: *const c_int, k: *const c_int,
                  alpha: *const c_float_complex, a: *const c_float_complex, lda: *const c_int,
                  beta: *const c_float_complex, c: *mut c_float_complex, ldc: *const c_int);
    pub fn cherk_(uplo: *const c_char, trans: *const c_char, n: *const c_int, k: *const c_int,
                  alpha: *const c_float, a: *const c_float_complex, lda: *const c_int,
                  beta: *const c_float, c: *mut c_float_complex, ldc: *const c_int);
    pub fn csyr2k_(uplo: *const c_char, trans: *const c_char, n: *const c_int, k: *const c_int,
                   alpha: *const c_float_complex, a: *const c_float_complex, lda: *const c_int,
                   b: *const c_float_complex, ldb: *const c_int, beta: *const c_float_complex,
                   c: *mut c_float_complex, ldc: *const c_int);
    pub fn cher2k_(uplo: *const c_char, trans: *const c_char, n: *const c_int, k: *const c_int,
                   alpha: *const c_float_complex, a: *const c_float_complex, lda: *const c_int,
                   b: *const c_float_complex, ldb: *const c_int, beta: *const c_float,
                   c: *mut c_float_complex, ldc: *const c_int);
    pub fn ctrmm_(side: *const c_char, uplo: *const c_char, transa: *const c_char,
                  diag: *const c_char, m: *const c_int, n: *const c_int,
                  alpha: *const c_float_complex, a: *const c_float_complex, lda: *const c_int,
                  b: *mut c_float_complex, ldb: *const c_int);
    pub fn ctrsm_(side: *const c_char, uplo: *const c_char, transa: *const c_char,
                  diag: *const c_char, m: *const c_int, n: *const c_int,
                  alpha: *const c_float_complex, a: *const c_float_complex, lda: *const c_int,
                  b: *mut c_float_complex, ldb: *const c_int);

    // Double complex
    pub fn zgemm_(transa: *const c_char, transb: *const c_char, m: *const c_int, n: *const c_int,
                  k: *const c_int, alpha: *const c_double_complex, a: *const c_double_complex,
                  lda: *const c_int, b: *const c_double_complex, ldb: *const c_int,
                  beta: *const c_double_complex, c: *mut c_double_complex, ldc: *const c_int);
    pub fn zsymm_(side: *const c_char, uplo: *const c_char, m: *const c_int, n: *const c_int,
                  alpha: *const c_double_complex, a: *const c_double_complex, lda: *const c_int,
                  b: *const c_double_complex, ldb: *const c_int, beta: *const c_double_complex,
                  c: *mut c_double_complex, ldc: *const c_int);
    pub fn zhemm_(side: *const c_char, uplo: *const c_char, m: *const c_int, n: *const c_int,
                  alpha: *const c_double_complex, a: *const c_double_complex, lda: *const c_int,
                  b: *const c_double_complex, ldb: *const c_int, beta: *const c_double_complex,
                  c: *mut c_double_complex, ldc: *const c_int);
    pub fn zsyrk_(uplo: *const c_char, trans: *const c_char, n: *const c_int, k: *const c_int,
                  alpha: *const c_double_complex, a: *const c_double_complex, lda: *const c_int,
                  beta: *const c_double_complex, c: *mut c_double_complex, ldc: *const c_int);
    pub fn zherk_(uplo: *const c_char, trans: *const c_char, n: *const c_int, k: *const c_int,
                  alpha: *const c_double, a: *const c_double_complex, lda: *const c_int,
                  beta: *const c_double, c: *mut c_double_complex, ldc: *const c_int);
    pub fn zsyr2k_(uplo: *const c_char, trans: *const c_char, n: *const c_int, k: *const c_int,
                   alpha: *const c_double_complex, a: *const c_double_complex, lda: *const c_int,
                   b: *const c_double_complex, ldb: *const c_int, beta: *const c_double_complex,
                   c: *mut c_double_complex, ldc: *const c_int);
    pub fn zher2k_(uplo: *const c_char, trans: *const c_char, n: *const c_int, k: *const c_int,
                   alpha: *const c_double_complex, a: *const c_double_complex, lda: *const c_int,
                   b: *const c_double_complex, ldb: *const c_int, beta: *const c_double,
                   c: *mut c_double_complex, ldc: *const c_int);
    pub fn ztrmm_(side: *const c_char, uplo: *const c_char, transa: *const c_char,
                  diag: *const c_char, m: *const c_int, n: *const c_int,
                  alpha: *const c_double_complex, a: *const c_double_complex, lda: *const c_int,
                  b: *mut c_double_complex, ldb: *const c_int);
    pub fn ztrsm_(side: *const c_char, uplo: *const c_char, transa: *const c_char,
                  diag: *const c_char, m: *const c_int, n: *const c_int,
                  alpha: *const c_double_complex, a: *const c_double_complex, lda: *const c_int,
                  b: *mut c_double_complex, ldb: *const c_int);
"""

def is_const(name, cty):
    return "*const" in cty

def is_mut(name, cty):
    return "*mut" in cty

def is_scalar(name, cty, f):
    return (
        "c_char" in cty or
        "c_int" in cty and (
            name in ["m", "n", "k", "kl", "ku"] or
            name.startswith("ld") or
            name.startswith("inc")
        ) or
        name in level_scalars[f.level]
    )

def translate_argument(name, cty, f):
    base = translate_type_base(cty)
    if is_const(name, cty):
        if is_scalar(name, cty, f):
            return base
        else:
            return "&[{}]".format(base)
    elif is_mut(name, cty):
        if is_scalar(name, cty, f):
            return "&mut {}".format(base)
        else:
            return "&mut [{}]".format(base)

    assert False, "cannot translate `{}: {}`".format(name, cty)

def translate_type_base(cty):
    if "c_char" in cty:
        return "u8"
    elif "c_int" in cty:
        return "i32"
    elif "c_double_complex" in cty:
        return "c64"
    elif "c_float_complex" in cty:
        return "c32"
    elif "double" in cty:
        return "f64"
    elif "float" in cty:
        return "f32"

    assert False, "cannot translate `{}`".format(cty)

def translate_body_argument(name, rty):
    if rty == "u8":
        return "&({} as c_char)".format(name)

    elif rty == "i32":
        return "&{}".format(name)

    elif rty.startswith("f"):
        return "&{}".format(name)
    elif rty.startswith("&mut f"):
        return "{}".format(name)
    elif rty.startswith("&[f"):
        return "{}.as_ptr()".format(name)
    elif rty.startswith("&mut [f"):
        return "{}.as_mut_ptr()".format(name)

    elif rty.startswith("c"):
        return "&{} as *const _ as *const _".format(name)
    elif rty.startswith("&mut c"):
        return "{} as *mut _ as *mut _".format(name)
    elif rty.startswith("&[c"):
        return "{}.as_ptr() as *const _".format(name)
    elif rty.startswith("&mut [c"):
        return "{}.as_mut_ptr() as *mut _".format(name)

    assert False, "cannot translate `{}: {}`".format(name, rty)

def translate_return_type(cty):
    if cty == "c_int":
        return "usize"
    elif cty == "c_float":
        return "f32"
    elif cty == "c_double":
        return "f64"

    assert False, "cannot translate `{}`".format(cty)

def format_header(f):
    args = format_header_arguments(f)
    ret = "" if f.ret is None else " -> {}".format(translate_return_type(f.ret))
    header = "pub fn {}({}){} {{".format(f.name, args, ret)

    s = []
    indent = 7 + len(f.name) + 1
    while True:
        if len(header) <= 99:
            s.append(header)
            break
        i = 98 - header[98::-1].index(',')
        if i < 0:
            s.append(header)
            break
        s.append(header[:i+1])
        header = "{}{}".format(" " * indent, header[i+2:])

    if len(s) > 1:
        s.append("")

    return "\n".join(s)

def format_body(f):
    a = format_body_arguments(f)
    r = format_body_return(f)
    if r is None:
        tail = "{})".format(a)
    else:
        tail = "{}) as {}".format(a, r)

    s = []
    s.append(" " * 4)
    s.append("unsafe {\n")
    s.append(" " * 8)
    s.append("ffi::{}_(".format(f.name))

    indent = 8 + 5 + len(f.name) + 2
    while len(tail) > 0:
        if len(tail) + indent > 99:
            i = tail.find(",")
            if i < 0 or i > 98:
                assert False, "cannot format `{}`".format(f.name)
            while True:
                l = tail.find(",", i + 1)
                if l < 0 or l + indent > 98: break
                i = l
            s.append(tail[0:i+1])
            s.append("\n")
            s.append(" " * indent)
            tail = tail[i+2:]
        else:
            s.append(tail)
            tail = ""

    s.append("\n")
    s.append(" " * 4)
    s.append("}")

    return "".join(s)

def format_header_arguments(f):
    s = []
    for arg in f.args:
        s.append("{}: {}".format(arg[0], translate_argument(*arg, f=f)))
    return ", ".join(s)

def format_body_arguments(f):
    s = []
    for arg in f.args:
        rty = translate_argument(*arg, f=f)
        s.append(translate_body_argument(arg[0], rty))
    return ", ".join(s)

def format_body_return(f):
    if f.ret is None:
        return None

    rty = translate_return_type(f.ret)
    if rty.startswith("f"):
        return None

    return rty

def prepare(level, code):
    lines = filter(lambda line: not re.match(r'^\s*//.*', line), code.split('\n'))
    lines = re.sub(r'\s+', ' ', "".join(lines)).strip().split(';')
    lines = filter(lambda line: not re.match(r'^\s*$', line), lines)
    return [Function.parse(level, line) for line in lines]

def do(funcs):
    for f in funcs:
        print("#[inline]")
        print(format_header(f))
        print(format_body(f))
        print("}\n")

if __name__ == "__main__":
    do(prepare(1, level1))
    do(prepare(2, level2))
    do(prepare(3, level3))
