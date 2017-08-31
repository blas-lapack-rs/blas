#!/usr/bin/env python

from common import Function
from documentation import print_documentation
import re, sys

level_scalars = {
    1: ["alpha", "a", "b", "c", "z"],
    2: ["alpha", "beta"],
    3: ["alpha", "beta"],
}

level1_functions = """
    pub fn cblas_dcabs1(z: *const c_double_complex) -> c_double;
    pub fn cblas_scabs1(c: *const c_float_complex) -> c_float;

    pub fn cblas_sdsdot(n: c_int, alpha: c_float, x: *const c_float, incx: c_int,
                        y: *const c_float, incy: c_int)
                        -> c_float;
    pub fn cblas_dsdot(n: c_int, x: *const c_float, incx: c_int, y: *const c_float, incy: c_int)
                       -> c_double;
    pub fn cblas_sdot(n: c_int, x: *const c_float, incx: c_int, y: *const c_float, incy: c_int)
                      -> c_float;
    pub fn cblas_ddot(n: c_int, x: *const c_double, incx: c_int, y: *const c_double, incy: c_int)
                      -> c_double;

    // Prefixes Z and C only
    pub fn cblas_cdotu_sub(n: c_int, x: *const c_float_complex, incx: c_int,
                           y: *const c_float_complex, incy: c_int, dotu: *mut c_float_complex);
    pub fn cblas_cdotc_sub(n: c_int, x: *const c_float_complex, incx: c_int,
                           y: *const c_float_complex, incy: c_int, dotc: *mut c_float_complex);

    pub fn cblas_zdotu_sub(n: c_int, x: *const c_double_complex, incx: c_int,
                           y: *const c_double_complex, incy: c_int, dotu: *mut c_double_complex);
    pub fn cblas_zdotc_sub(n: c_int, x: *const c_double_complex, incx: c_int,
                           y: *const c_double_complex, incy: c_int, dotc: *mut c_double_complex);

    // Prefixes S, D, SC, and DZ
    pub fn cblas_snrm2(n: c_int, x: *const c_float, incx: c_int) -> c_float;
    pub fn cblas_sasum(n: c_int, x: *const c_float, incx: c_int) -> c_float;

    pub fn cblas_dnrm2(n: c_int, x: *const c_double, incx: c_int) -> c_double;
    pub fn cblas_dasum(n: c_int, x: *const c_double, incx: c_int) -> c_double;

    pub fn cblas_scnrm2(n: c_int, x: *const c_float_complex, incx: c_int) -> c_float;
    pub fn cblas_scasum(n: c_int, x: *const c_float_complex, incx: c_int) -> c_float;

    pub fn cblas_dznrm2(n: c_int, x: *const c_double_complex, incx: c_int) -> c_double;
    pub fn cblas_dzasum(n: c_int, x: *const c_double_complex, incx: c_int) -> c_double;

    // Standard prefixes (S, D, C, and Z)
    pub fn cblas_isamax(n: c_int, x: *const c_float, incx: c_int) -> CBLAS_INDEX;
    pub fn cblas_idamax(n: c_int, x: *const c_double, incx: c_int) -> CBLAS_INDEX;
    pub fn cblas_icamax(n: c_int, x: *const c_float_complex, incx: c_int) -> CBLAS_INDEX;
    pub fn cblas_izamax(n: c_int, x: *const c_double_complex, incx: c_int) -> CBLAS_INDEX;
"""

level1_routines = """
    // Standard prefixes (S, D, C, and Z)
    pub fn cblas_sswap(n: c_int, x: *mut c_float, incx: c_int, y: *mut c_float, incy: c_int);
    pub fn cblas_scopy(n: c_int, x: *const c_float, incx: c_int, y: *mut c_float, incy: c_int);
    pub fn cblas_saxpy(n: c_int, alpha: c_float, x: *const c_float, incx: c_int,
                       y: *mut c_float, incy: c_int);

    pub fn cblas_dswap(n: c_int, x: *mut c_double, incx: c_int, y: *mut c_double, incy: c_int);
    pub fn cblas_dcopy(n: c_int, x: *const c_double, incx: c_int, y: *mut c_double, incy: c_int);
    pub fn cblas_daxpy(n: c_int, alpha: c_double, x: *const c_double, incx: c_int,
                       y: *mut c_double, incy: c_int);

    pub fn cblas_cswap(n: c_int, x: *mut c_float_complex, incx: c_int, y: *mut c_float_complex,
                       incy: c_int);
    pub fn cblas_ccopy(n: c_int, x: *const c_float_complex, incx: c_int,
                       y: *mut c_float_complex, incy: c_int);
    pub fn cblas_caxpy(n: c_int, alpha: *const c_float_complex, x: *const c_float_complex,
                       incx: c_int, y: *mut c_float_complex, incy: c_int);

    pub fn cblas_zswap(n: c_int, x: *mut c_double_complex, incx: c_int,
                       y: *mut c_double_complex, incy: c_int);
    pub fn cblas_zcopy(n: c_int, x: *const c_double_complex, incx: c_int,
                       y: *mut c_double_complex, incy: c_int);
    pub fn cblas_zaxpy(n: c_int, alpha: *const c_double_complex, x: *const c_double_complex,
                       incx: c_int, y: *mut c_double_complex, incy: c_int);

    // Prefixes S and D only
    pub fn cblas_srotg(a: *mut c_float, b: *mut c_float, c: *mut c_float, s: *mut c_float);
    pub fn cblas_srotmg(d1: *mut c_float, d2: *mut c_float, b1: *mut c_float, b2: c_float,
                        p: *mut c_float);
    pub fn cblas_srot(n: c_int, x: *mut c_float, incx: c_int, y: *mut c_float, incy: c_int,
                      c: c_float, s: c_float);
    pub fn cblas_srotm(n: c_int, x: *mut c_float, incx: c_int, y: *mut c_float, incy: c_int,
                       p: *const c_float);

    pub fn cblas_drotg(a: *mut c_double, b: *mut c_double, c: *mut c_double, s: *mut c_double);
    pub fn cblas_drotmg(d1: *mut c_double, d2: *mut c_double, b1: *mut c_double, b2: c_double,
                        p: *mut c_double);
    pub fn cblas_drot(n: c_int, x: *mut c_double, incx: c_int, y: *mut c_double, incy: c_int,
                      c: c_double, s: c_double);
    pub fn cblas_drotm(n: c_int, x: *mut c_double, incx: c_int, y: *mut c_double, incy: c_int,
                       p: *const c_double);

    // Prefixes S, D, C, Z, CS, and ZD
    pub fn cblas_sscal(n: c_int, alpha: c_float, x: *mut c_float, incx: c_int);
    pub fn cblas_dscal(n: c_int, alpha: c_double, x: *mut c_double, incx: c_int);
    pub fn cblas_cscal(n: c_int, alpha: *const c_float_complex, x: *mut c_float_complex,
                       incx: c_int);
    pub fn cblas_zscal(n: c_int, alpha: *const c_double_complex, x: *mut c_double_complex,
                       incx: c_int);
    pub fn cblas_csscal(n: c_int, alpha: c_float, x: *mut c_float_complex, incx: c_int);
    pub fn cblas_zdscal(n: c_int, alpha: c_double, x: *mut c_double_complex, incx: c_int);
"""

level2 = """
    // Standard prefixes (S, D, C, and Z)
    pub fn cblas_sgemv(layout: CBLAS_LAYOUT, transa: CBLAS_TRANSPOSE, m: c_int, n: c_int,
                       alpha: c_float, a: *const c_float, lda: c_int, x: *const c_float,
                       incx: c_int, beta: c_float, y: *mut c_float, incy: c_int);
    pub fn cblas_sgbmv(layout: CBLAS_LAYOUT, transa: CBLAS_TRANSPOSE, m: c_int, n: c_int,
                       kl: c_int, ku: c_int, alpha: c_float, a: *const c_float, lda: c_int,
                       x: *const c_float, incx: c_int, beta: c_float, y: *mut c_float,
                       incy: c_int);
    pub fn cblas_strmv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, a: *const c_float, lda: c_int,
                       x: *mut c_float, incx: c_int);
    pub fn cblas_stbmv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, k: c_int, a: *const c_float, lda: c_int,
                       x: *mut c_float, incx: c_int);
    pub fn cblas_stpmv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, ap: *const c_float, x: *mut c_float,
                       incx: c_int);
    pub fn cblas_strsv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, a: *const c_float, lda: c_int,
                       x: *mut c_float, incx: c_int);
    pub fn cblas_stbsv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, k: c_int, a: *const c_float, lda: c_int,
                       x: *mut c_float, incx: c_int);
    pub fn cblas_stpsv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, ap: *const c_float, x: *mut c_float,
                       incx: c_int);

    pub fn cblas_dgemv(layout: CBLAS_LAYOUT, transa: CBLAS_TRANSPOSE, m: c_int, n: c_int,
                       alpha: c_double, a: *const c_double, lda: c_int, x: *const c_double,
                       incx: c_int, beta: c_double, y: *mut c_double, incy: c_int);
    pub fn cblas_dgbmv(layout: CBLAS_LAYOUT, transa: CBLAS_TRANSPOSE, m: c_int, n: c_int,
                       kl: c_int, ku: c_int, alpha: c_double, a: *const c_double, lda: c_int,
                       x: *const c_double, incx: c_int, beta: c_double, y: *mut c_double,
                       incy: c_int);
    pub fn cblas_dtrmv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, a: *const c_double, lda: c_int,
                       x: *mut c_double, incx: c_int);
    pub fn cblas_dtbmv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, k: c_int, a: *const c_double, lda: c_int,
                       x: *mut c_double, incx: c_int);
    pub fn cblas_dtpmv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, ap: *const c_double, x: *mut c_double,
                       incx: c_int);
    pub fn cblas_dtrsv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, a: *const c_double, lda: c_int,
                       x: *mut c_double, incx: c_int);
    pub fn cblas_dtbsv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, k: c_int, a: *const c_double, lda: c_int,
                       x: *mut c_double, incx: c_int);
    pub fn cblas_dtpsv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, ap: *const c_double, x: *mut c_double,
                       incx: c_int);

    pub fn cblas_cgemv(layout: CBLAS_LAYOUT, transa: CBLAS_TRANSPOSE, m: c_int, n: c_int,
                       alpha: *const c_float_complex, a: *const c_float_complex, lda: c_int,
                       x: *const c_float_complex, incx: c_int, beta: *const c_float_complex,
                       y: *mut c_float_complex, incy: c_int);
    pub fn cblas_cgbmv(layout: CBLAS_LAYOUT, transa: CBLAS_TRANSPOSE, m: c_int, n: c_int,
                       kl: c_int, ku: c_int, alpha: *const c_float_complex,
                       a: *const c_float_complex, lda: c_int, x: *const c_float_complex,
                       incx: c_int, beta: *const c_float_complex, y: *mut c_float_complex,
                       incy: c_int);
    pub fn cblas_ctrmv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, a: *const c_float_complex, lda: c_int,
                       x: *mut c_float_complex, incx: c_int);
    pub fn cblas_ctbmv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, k: c_int, a: *const c_float_complex,
                       lda: c_int, x: *mut c_float_complex, incx: c_int);
    pub fn cblas_ctpmv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, ap: *const c_float_complex,
                       x: *mut c_float_complex, incx: c_int);
    pub fn cblas_ctrsv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, a: *const c_float_complex, lda: c_int,
                       x: *mut c_float_complex, incx: c_int);
    pub fn cblas_ctbsv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, k: c_int, a: *const c_float_complex,
                       lda: c_int, x: *mut c_float_complex, incx: c_int);
    pub fn cblas_ctpsv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, ap: *const c_float_complex,
                       x: *mut c_float_complex, incx: c_int);

    pub fn cblas_zgemv(layout: CBLAS_LAYOUT, transa: CBLAS_TRANSPOSE, m: c_int, n: c_int,
                       alpha: *const c_double_complex, a: *const c_double_complex, lda: c_int,
                       x: *const c_double_complex, incx: c_int, beta: *const c_double_complex,
                       y: *mut c_double_complex, incy: c_int);
    pub fn cblas_zgbmv(layout: CBLAS_LAYOUT, transa: CBLAS_TRANSPOSE, m: c_int, n: c_int,
                       kl: c_int, ku: c_int, alpha: *const c_double_complex,
                       a: *const c_double_complex, lda: c_int, x: *const c_double_complex,
                       incx: c_int, beta: *const c_double_complex, y: *mut c_double_complex,
                       incy: c_int);
    pub fn cblas_ztrmv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, a: *const c_double_complex, lda: c_int,
                       x: *mut c_double_complex, incx: c_int);
    pub fn cblas_ztbmv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, k: c_int, a: *const c_double_complex,
                       lda: c_int, x: *mut c_double_complex, incx: c_int);
    pub fn cblas_ztpmv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, ap: *const c_double_complex,
                       x: *mut c_double_complex, incx: c_int);
    pub fn cblas_ztrsv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, a: *const c_double_complex, lda: c_int,
                       x: *mut c_double_complex, incx: c_int);
    pub fn cblas_ztbsv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, k: c_int, a: *const c_double_complex,
                       lda: c_int, x: *mut c_double_complex, incx: c_int);
    pub fn cblas_ztpsv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, transa: CBLAS_TRANSPOSE,
                       diag: CBLAS_DIAG, n: c_int, ap: *const c_double_complex,
                       x: *mut c_double_complex, incx: c_int);

    // Prefixes S and D only
    pub fn cblas_ssymv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int, alpha: c_float,
                       a: *const c_float, lda: c_int, x: *const c_float, incx: c_int,
                       beta: c_float, y: *mut c_float, incy: c_int);
    pub fn cblas_ssbmv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int, k: c_int,
                       alpha: c_float, a: *const c_float, lda: c_int, x: *const c_float,
                       incx: c_int, beta: c_float, y: *mut c_float, incy: c_int);
    pub fn cblas_sspmv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int, alpha: c_float,
                       ap: *const c_float, x: *const c_float, incx: c_int, beta: c_float,
                       y: *mut c_float, incy: c_int);
    pub fn cblas_sger(layout: CBLAS_LAYOUT, m: c_int, n: c_int, alpha: c_float,
                      x: *const c_float, incx: c_int, y: *const c_float, incy: c_int,
                      a: *mut c_float, lda: c_int);
    pub fn cblas_ssyr(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int, alpha: c_float,
                      x: *const c_float, incx: c_int, a: *mut c_float, lda: c_int);
    pub fn cblas_sspr(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int, alpha: c_float,
                      x: *const c_float, incx: c_int, ap: *mut c_float);
    pub fn cblas_ssyr2(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int, alpha: c_float,
                       x: *const c_float, incx: c_int, y: *const c_float, incy: c_int,
                       a: *mut c_float, lda: c_int);
    pub fn cblas_sspr2(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int, alpha: c_float,
                       x: *const c_float, incx: c_int, y: *const c_float, incy: c_int,
                       a: *mut c_float);

    pub fn cblas_dsymv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int, alpha: c_double,
                       a: *const c_double, lda: c_int, x: *const c_double, incx: c_int,
                       beta: c_double, y: *mut c_double, incy: c_int);
    pub fn cblas_dsbmv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int, k: c_int,
                       alpha: c_double, a: *const c_double, lda: c_int, x: *const c_double,
                       incx: c_int, beta: c_double, y: *mut c_double, incy: c_int);
    pub fn cblas_dspmv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int, alpha: c_double,
                       ap: *const c_double, x: *const c_double, incx: c_int, beta: c_double,
                       y: *mut c_double, incy: c_int);
    pub fn cblas_dger(layout: CBLAS_LAYOUT, m: c_int, n: c_int, alpha: c_double,
                      x: *const c_double, incx: c_int, y: *const c_double, incy: c_int,
                      a: *mut c_double, lda: c_int);
    pub fn cblas_dsyr(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int, alpha: c_double,
                      x: *const c_double, incx: c_int, a: *mut c_double, lda: c_int);
    pub fn cblas_dspr(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int, alpha: c_double,
                      x: *const c_double, incx: c_int, ap: *mut c_double);
    pub fn cblas_dsyr2(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int, alpha: c_double,
                       x: *const c_double, incx: c_int, y: *const c_double, incy: c_int,
                       a: *mut c_double, lda: c_int);
    pub fn cblas_dspr2(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int, alpha: c_double,
                       x: *const c_double, incx: c_int, y: *const c_double, incy: c_int,
                       a: *mut c_double);

    // Prefixes C and Z only
    pub fn cblas_chemv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int,
                       alpha: *const c_float_complex, a: *const c_float_complex, lda: c_int,
                       x: *const c_float_complex, incx: c_int, beta: *const c_float_complex,
                       y: *mut c_float_complex, incy: c_int);
    pub fn cblas_chbmv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int, k: c_int,
                       alpha: *const c_float_complex, a: *const c_float_complex, lda: c_int,
                       x: *const c_float_complex, incx: c_int, beta: *const c_float_complex,
                       y: *mut c_float_complex, incy: c_int);
    pub fn cblas_chpmv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int,
                       alpha: *const c_float_complex, ap: *const c_float_complex,
                       x: *const c_float_complex, incx: c_int, beta: *const c_float_complex,
                       y: *mut c_float_complex, incy: c_int);
    pub fn cblas_cgeru(layout: CBLAS_LAYOUT, m: c_int, n: c_int, alpha: *const c_float_complex,
                       x: *const c_float_complex, incx: c_int, y: *const c_float_complex,
                       incy: c_int, a: *mut c_float_complex, lda: c_int);
    pub fn cblas_cgerc(layout: CBLAS_LAYOUT, m: c_int, n: c_int, alpha: *const c_float_complex,
                       x: *const c_float_complex, incx: c_int, y: *const c_float_complex,
                       incy: c_int, a: *mut c_float_complex, lda: c_int);
    pub fn cblas_cher(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int, alpha: c_float,
                      x: *const c_float_complex, incx: c_int, a: *mut c_float_complex,
                      lda: c_int);
    pub fn cblas_chpr(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int, alpha: c_float,
                      x: *const c_float_complex, incx: c_int, a: *mut c_float_complex);
    pub fn cblas_cher2(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int,
                       alpha: *const c_float_complex, x: *const c_float_complex, incx: c_int,
                       y: *const c_float_complex, incy: c_int, a: *mut c_float_complex,
                       lda: c_int);
    pub fn cblas_chpr2(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int,
                       alpha: *const c_float_complex, x: *const c_float_complex, incx: c_int,
                       y: *const c_float_complex, incy: c_int, ap: *mut c_float_complex);

    pub fn cblas_zhemv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int,
                       alpha: *const c_double_complex, a: *const c_double_complex, lda: c_int,
                       x: *const c_double_complex, incx: c_int, beta: *const c_double_complex,
                       y: *mut c_double_complex, incy: c_int);
    pub fn cblas_zhbmv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int, k: c_int,
                       alpha: *const c_double_complex, a: *const c_double_complex, lda: c_int,
                       x: *const c_double_complex, incx: c_int, beta: *const c_double_complex,
                       y: *mut c_double_complex, incy: c_int);
    pub fn cblas_zhpmv(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int,
                       alpha: *const c_double_complex, ap: *const c_double_complex,
                       x: *const c_double_complex, incx: c_int, beta: *const c_double_complex,
                       y: *mut c_double_complex, incy: c_int);
    pub fn cblas_zgeru(layout: CBLAS_LAYOUT, m: c_int, n: c_int, alpha: *const c_double_complex,
                       x: *const c_double_complex, incx: c_int, y: *const c_double_complex,
                       incy: c_int, a: *mut c_double_complex, lda: c_int);
    pub fn cblas_zgerc(layout: CBLAS_LAYOUT, m: c_int, n: c_int, alpha: *const c_double_complex,
                       x: *const c_double_complex, incx: c_int, y: *const c_double_complex,
                       incy: c_int, a: *mut c_double_complex, lda: c_int);
    pub fn cblas_zher(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int, alpha: c_double,
                      x: *const c_double_complex, incx: c_int, a: *mut c_double_complex,
                      lda: c_int);
    pub fn cblas_zhpr(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int, alpha: c_double,
                      x: *const c_double_complex, incx: c_int, a: *mut c_double_complex);
    pub fn cblas_zher2(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int,
                       alpha: *const c_double_complex, x: *const c_double_complex, incx: c_int,
                       y: *const c_double_complex, incy: c_int, a: *mut c_double_complex,
                       lda: c_int);
    pub fn cblas_zhpr2(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, n: c_int,
                       alpha: *const c_double_complex, x: *const c_double_complex, incx: c_int,
                       y: *const c_double_complex, incy: c_int, ap: *mut c_double_complex);
"""

level3 = """
    // Standard prefixes (S, D, C, and Z)
    pub fn cblas_sgemm(layout: CBLAS_LAYOUT, transa: CBLAS_TRANSPOSE, transb: CBLAS_TRANSPOSE,
                       m: c_int, n: c_int, k: c_int, alpha: c_float, a: *const c_float,
                       lda: c_int, b: *const c_float, ldb: c_int, beta: c_float,
                       c: *mut c_float, ldc: c_int);
    pub fn cblas_ssymm(layout: CBLAS_LAYOUT, side: CBLAS_SIDE, uplo: CBLAS_UPLO, m: c_int,
                       n: c_int, alpha: c_float, a: *const c_float, lda: c_int,
                       b: *const c_float, ldb: c_int, beta: c_float, c: *mut c_float, ldc: c_int);
    pub fn cblas_ssyrk(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE, n: c_int,
                       k: c_int, alpha: c_float, a: *const c_float, lda: c_int, beta: c_float,
                       c: *mut c_float, ldc: c_int);
    pub fn cblas_ssyr2k(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE,
                        n: c_int, k: c_int, alpha: c_float, a: *const c_float, lda: c_int,
                        b: *const c_float, ldb: c_int, beta: c_float, c: *mut c_float,
                        ldc: c_int);
    pub fn cblas_strmm(layout: CBLAS_LAYOUT, side: CBLAS_SIDE, uplo: CBLAS_UPLO,
                       transa: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, m: c_int, n: c_int,
                       alpha: c_float, a: *const c_float, lda: c_int, b: *mut c_float,
                       ldb: c_int);
    pub fn cblas_strsm(layout: CBLAS_LAYOUT, side: CBLAS_SIDE, uplo: CBLAS_UPLO,
                       transa: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, m: c_int, n: c_int,
                       alpha: c_float, a: *const c_float, lda: c_int, b: *mut c_float,
                       ldb: c_int);

    pub fn cblas_dgemm(layout: CBLAS_LAYOUT, transa: CBLAS_TRANSPOSE, transb: CBLAS_TRANSPOSE,
                       m: c_int, n: c_int, k: c_int, alpha: c_double, a: *const c_double,
                       lda: c_int, b: *const c_double, ldb: c_int, beta: c_double,
                       c: *mut c_double, ldc: c_int);
    pub fn cblas_dsymm(layout: CBLAS_LAYOUT, side: CBLAS_SIDE, uplo: CBLAS_UPLO, m: c_int,
                       n: c_int, alpha: c_double, a: *const c_double, lda: c_int,
                       b: *const c_double, ldb: c_int, beta: c_double, c: *mut c_double,
                       ldc: c_int);
    pub fn cblas_dsyrk(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE, n: c_int,
                       k: c_int, alpha: c_double, a: *const c_double, lda: c_int,
                       beta: c_double, c: *mut c_double, ldc: c_int);
    pub fn cblas_dsyr2k(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE,
                        n: c_int, k: c_int, alpha: c_double, a: *const c_double, lda: c_int,
                        b: *const c_double, ldb: c_int, beta: c_double, c: *mut c_double,
                        ldc: c_int);
    pub fn cblas_dtrmm(layout: CBLAS_LAYOUT, side: CBLAS_SIDE, uplo: CBLAS_UPLO,
                       transa: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, m: c_int, n: c_int,
                       alpha: c_double, a: *const c_double, lda: c_int, b: *mut c_double,
                       ldb: c_int);
    pub fn cblas_dtrsm(layout: CBLAS_LAYOUT, side: CBLAS_SIDE, uplo: CBLAS_UPLO,
                       transa: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, m: c_int, n: c_int,
                       alpha: c_double, a: *const c_double, lda: c_int, b: *mut c_double,
                       ldb: c_int);

    pub fn cblas_cgemm(layout: CBLAS_LAYOUT, transa: CBLAS_TRANSPOSE, transb: CBLAS_TRANSPOSE,
                       m: c_int, n: c_int, k: c_int, alpha: *const c_float_complex,
                       a: *const c_float_complex, lda: c_int, b: *const c_float_complex,
                       ldb: c_int, beta: *const c_float_complex, c: *mut c_float_complex,
                       ldc: c_int);
    pub fn cblas_csymm(layout: CBLAS_LAYOUT, side: CBLAS_SIDE, uplo: CBLAS_UPLO, m: c_int,
                       n: c_int, alpha: *const c_float_complex, a: *const c_float_complex,
                       lda: c_int, b: *const c_float_complex, ldb: c_int,
                       beta: *const c_float_complex, c: *mut c_float_complex, ldc: c_int);
    pub fn cblas_csyrk(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE, n: c_int,
                       k: c_int, alpha: *const c_float_complex, a: *const c_float_complex,
                       lda: c_int, beta: *const c_float_complex, c: *mut c_float_complex,
                       ldc: c_int);
    pub fn cblas_csyr2k(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE,
                        n: c_int, k: c_int, alpha: *const c_float_complex,
                        a: *const c_float_complex, lda: c_int, b: *const c_float_complex,
                        ldb: c_int, beta: *const c_float_complex, c: *mut c_float_complex,
                        ldc: c_int);
    pub fn cblas_ctrmm(layout: CBLAS_LAYOUT, side: CBLAS_SIDE, uplo: CBLAS_UPLO,
                       transa: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, m: c_int, n: c_int,
                       alpha: *const c_float_complex, a: *const c_float_complex, lda: c_int,
                       b: *mut c_float_complex, ldb: c_int);
    pub fn cblas_ctrsm(layout: CBLAS_LAYOUT, side: CBLAS_SIDE, uplo: CBLAS_UPLO,
                       transa: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, m: c_int, n: c_int,
                       alpha: *const c_float_complex, a: *const c_float_complex, lda: c_int,
                       b: *mut c_float_complex, ldb: c_int);

    pub fn cblas_zgemm(layout: CBLAS_LAYOUT, transa: CBLAS_TRANSPOSE, transb: CBLAS_TRANSPOSE,
                       m: c_int, n: c_int, k: c_int, alpha: *const c_double_complex,
                       a: *const c_double_complex, lda: c_int, b: *const c_double_complex,
                       ldb: c_int, beta: *const c_double_complex, c: *mut c_double_complex,
                       ldc: c_int);
    pub fn cblas_zsymm(layout: CBLAS_LAYOUT, side: CBLAS_SIDE, uplo: CBLAS_UPLO, m: c_int,
                       n: c_int, alpha: *const c_double_complex, a: *const c_double_complex,
                       lda: c_int, b: *const c_double_complex, ldb: c_int,
                       beta: *const c_double_complex, c: *mut c_double_complex, ldc: c_int);
    pub fn cblas_zsyrk(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE, n: c_int,
                       k: c_int, alpha: *const c_double_complex, a: *const c_double_complex,
                       lda: c_int, beta: *const c_double_complex, c: *mut c_double_complex,
                       ldc: c_int);
    pub fn cblas_zsyr2k(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE,
                        n: c_int, k: c_int, alpha: *const c_double_complex,
                        a: *const c_double_complex, lda: c_int, b: *const c_double_complex,
                        ldb: c_int, beta: *const c_double_complex, c: *mut c_double_complex,
                        ldc: c_int);
    pub fn cblas_ztrmm(layout: CBLAS_LAYOUT, side: CBLAS_SIDE, uplo: CBLAS_UPLO,
                       transa: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, m: c_int, n: c_int,
                       alpha: *const c_double_complex, a: *const c_double_complex, lda: c_int,
                       b: *mut c_double_complex, ldb: c_int);
    pub fn cblas_ztrsm(layout: CBLAS_LAYOUT, side: CBLAS_SIDE, uplo: CBLAS_UPLO,
                       transa: CBLAS_TRANSPOSE, diag: CBLAS_DIAG, m: c_int, n: c_int,
                       alpha: *const c_double_complex, a: *const c_double_complex, lda: c_int,
                       b: *mut c_double_complex, ldb: c_int);

    // Prefixes C and Z only
    pub fn cblas_chemm(layout: CBLAS_LAYOUT, side: CBLAS_SIDE, uplo: CBLAS_UPLO, m: c_int,
                       n: c_int, alpha: *const c_float_complex, a: *const c_float_complex,
                       lda: c_int, b: *const c_float_complex, ldb: c_int,
                       beta: *const c_float_complex, c: *mut c_float_complex, ldc: c_int);
    pub fn cblas_cherk(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE, n: c_int,
                       k: c_int, alpha: c_float, a: *const c_float_complex, lda: c_int,
                       beta: c_float, c: *mut c_float_complex, ldc: c_int);
    pub fn cblas_cher2k(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE,
                        n: c_int, k: c_int, alpha: *const c_float_complex,
                        a: *const c_float_complex, lda: c_int, b: *const c_float_complex,
                        ldb: c_int, beta: c_float, c: *mut c_float_complex, ldc: c_int);

    pub fn cblas_zhemm(layout: CBLAS_LAYOUT, side: CBLAS_SIDE, uplo: CBLAS_UPLO, m: c_int,
                       n: c_int, alpha: *const c_double_complex, a: *const c_double_complex,
                       lda: c_int, b: *const c_double_complex, ldb: c_int,
                       beta: *const c_double_complex, c: *mut c_double_complex, ldc: c_int);
    pub fn cblas_zherk(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE, n: c_int,
                       k: c_int, alpha: c_double, a: *const c_double_complex, lda: c_int,
                       beta: c_double, c: *mut c_double_complex, ldc: c_int);
    pub fn cblas_zher2k(layout: CBLAS_LAYOUT, uplo: CBLAS_UPLO, trans: CBLAS_TRANSPOSE,
                        n: c_int, k: c_int, alpha: *const c_double_complex,
                        a: *const c_double_complex, lda: c_int, b: *const c_double_complex,
                        ldb: c_int, beta: c_double, c: *mut c_double_complex, ldc: c_int);
"""

def translate_argument(name, cty, f):
    if cty == "c_int":
        return "i32"

    elif cty == "CBLAS_DIAG":
        return "Diagonal"
    elif cty == "CBLAS_LAYOUT":
        return "Layout"
    elif cty == "CBLAS_SIDE":
        return "Side"
    elif cty == "CBLAS_TRANSPOSE":
        return "Transpose"
    elif cty == "CBLAS_UPLO":
        return "Part"

    base = translate_type_base(cty)

    if "*const" in cty:
        if is_scalar(name, cty, f):
            return base
        else:
            return "&[{}]".format(base)
    elif "*mut" in cty:
        if is_scalar(name, cty, f):
            return "&mut {}".format(base)
        else:
            return "&mut [{}]".format(base)

    return base

def is_scalar(name, cty, f):
    return name in level_scalars[f.level]

def translate_type_base(cty):
    if "c_double_complex" in cty:
        return "c64"
    elif "c_float_complex" in cty:
        return "c32"
    elif "double" in cty:
        return "f64"
    elif "float" in cty:
        return "f32"

    assert False, "cannot translate `{}`".format(cty)

def translate_body_argument(name, rty):
    if rty == "i32":
        return name

    elif rty in ["Diagonal", "Layout", "Part", "Side", "Transpose"]:
        return "{}.into()".format(name)

    elif rty.startswith("f"):
        return name
    elif rty.startswith("&mut f"):
        return name
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
        return "i32"
    elif cty == "c_float":
        return "f32"
    elif cty == "c_double":
        return "f64"

    if cty == "CBLAS_INDEX":
        return "i32"

    assert False, "cannot translate `{}`".format(cty)

def format_header(f):
    args = format_header_arguments(f)
    if f.ret is None:
        return "pub unsafe fn {}({})".format(f.name, args)
    else:
        return "pub unsafe fn {}({}) -> {}".format(f.name, args, translate_return_type(f.ret))

def format_body(f):
    args = format_body_arguments(f)
    ret = format_body_return(f)
    if ret is None:
        return "ffi::cblas_{}({})".format(f.name, args)
    else:
        return "ffi::cblas_{}({}) as {}".format(f.name, args, ret)

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

def do(functions, reference):
    for f in functions:
        if reference is not None:
            print_documentation(f, reference)
        print("\n#[inline]")
        print(format_header(f) + " {")
        print("    " + format_body(f) + "\n}")

reference = sys.argv[1] if len(sys.argv) > 1 else None
do(prepare(1, level1_functions), reference)
do(prepare(1, level1_routines), reference)
do(prepare(2, level2), reference)
do(prepare(3, level3), reference)
