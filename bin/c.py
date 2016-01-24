#!/usr/bin/env python

import re

level_scalars = {
    1: ["c", "z"],
    2: [],
    3: [],
    0: [],
}

def is_scalar(name, cty, f):
    return name in level_scalars[f.level]

level1_functions = """
    pub fn cblas_dcabs1(z: *const c_double_complex) -> c_double;
    pub fn cblas_scabs1(c: *const c_float_complex) -> c_float;

    pub fn cblas_sdsdot(n: c_int, alpha: c_float, x: *const c_float, incx: c_int,
                        y: *const c_float, incy: c_int) -> c_float;
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
    pub fn cblas_isamax(n: c_int, x: *const c_float, incx: c_int) -> CblasIndex;
    pub fn cblas_idamax(n: c_int, x: *const c_double, incx: c_int) -> CblasIndex;
    pub fn cblas_icamax(n: c_int, x: *const c_float_complex, incx: c_int) -> CblasIndex;
    pub fn cblas_izamax(n: c_int, x: *const c_double_complex, incx: c_int) -> CblasIndex;
"""

level1_routines = """
    // Standard prefixes (S, D, C, and Z)
    pub fn cblas_sswap(n: c_int, x: *mut c_float, incx: c_int, y: *mut c_float, incy: c_int);
    pub fn cblas_scopy(n: c_int, x: *const c_float, incx: c_int, y: *mut c_float, incy: c_int);
    pub fn cblas_saxpy(n: c_int, alpha: c_float, x: *const c_float, incx: c_int, y: *mut c_float,
                       incy: c_int);

    pub fn cblas_dswap(n: c_int, x: *mut c_double, incx: c_int, y: *mut c_double, incy: c_int);
    pub fn cblas_dcopy(n: c_int, x: *const c_double, incx: c_int, y: *mut c_double, incy: c_int);
    pub fn cblas_daxpy(n: c_int, alpha: c_double, x: *const c_double, incx: c_int,
                       y: *mut c_double, incy: c_int);

    pub fn cblas_cswap(n: c_int, x: *mut c_float_complex, incx: c_int, y: *mut c_float_complex,
                       incy: c_int);
    pub fn cblas_ccopy(n: c_int, x: *const c_float_complex, incx: c_int, y: *mut c_float_complex,
                       incy: c_int);
    pub fn cblas_caxpy(n: c_int, alpha: *const c_float_complex, x: *const c_float_complex,
                       incx: c_int, y: *mut c_float_complex, incy: c_int);

    pub fn cblas_zswap(n: c_int, x: *mut c_double_complex, incx: c_int, y: *mut c_double_complex,
                       incy: c_int);
    pub fn cblas_zcopy(n: c_int, x: *const c_double_complex, incx: c_int, y: *mut c_double_complex,
                       incy: c_int);
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
    pub fn cblas_sgemv(layout: CblasLayout, transa: CblasTranspose, m: c_int, n: c_int,
                       alpha: c_float, a: *const c_float, lda: c_int, x: *const c_float,
                       incx: c_int, beta: c_float, y: *mut c_float, incy: c_int);
    pub fn cblas_sgbmv(layout: CblasLayout, transa: CblasTranspose, m: c_int, n: c_int, kl: c_int,
                       ku: c_int, alpha: c_float, a: *const c_float, lda: c_int, x: *const c_float,
                       incx: c_int, beta: c_float, y: *mut c_float, incy: c_int);
    pub fn cblas_strmv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, a: *const c_float, lda: c_int, x: *mut c_float,
                       incx: c_int);
    pub fn cblas_stbmv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, k: c_int, a: *const c_float, lda: c_int,
                       x: *mut c_float, incx: c_int);
    pub fn cblas_stpmv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, ap: *const c_float, x: *mut c_float,
                       incx: c_int);
    pub fn cblas_strsv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, a: *const c_float, lda: c_int, x: *mut c_float,
                       incx: c_int);
    pub fn cblas_stbsv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, k: c_int, a: *const c_float, lda: c_int,
                       x: *mut c_float, incx: c_int);
    pub fn cblas_stpsv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, ap: *const c_float, x: *mut c_float,
                       incx: c_int);

    pub fn cblas_dgemv(layout: CblasLayout, transa: CblasTranspose, m: c_int, n: c_int,
                       alpha: c_double, a: *const c_double, lda: c_int, x: *const c_double,
                       incx: c_int, beta: c_double, y: *mut c_double, incy: c_int);
    pub fn cblas_dgbmv(layout: CblasLayout, transa: CblasTranspose, m: c_int, n: c_int, kl: c_int,
                       ku: c_int, alpha: c_double, a: *const c_double, lda: c_int,
                       x: *const c_double, incx: c_int, beta: c_double, y: *mut c_double,
                       incy: c_int);
    pub fn cblas_dtrmv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, a: *const c_double, lda: c_int, x: *mut c_double,
                       incx: c_int);
    pub fn cblas_dtbmv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, k: c_int, a: *const c_double, lda: c_int,
                       x: *mut c_double, incx: c_int);
    pub fn cblas_dtpmv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, ap: *const c_double, x: *mut c_double,
                       incx: c_int);
    pub fn cblas_dtrsv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, a: *const c_double, lda: c_int, x: *mut c_double,
                       incx: c_int);
    pub fn cblas_dtbsv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, k: c_int, a: *const c_double, lda: c_int,
                       x: *mut c_double, incx: c_int);
    pub fn cblas_dtpsv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, ap: *const c_double, x: *mut c_double,
                       incx: c_int);

    pub fn cblas_cgemv(layout: CblasLayout, transa: CblasTranspose, m: c_int, n: c_int,
                       alpha: *const c_float_complex, a: *const c_float_complex, lda: c_int,
                       x: *const c_float_complex, incx: c_int, beta: *const c_float_complex,
                       y: *mut c_float_complex, incy: c_int);
    pub fn cblas_cgbmv(layout: CblasLayout, transa: CblasTranspose, m: c_int, n: c_int, kl: c_int,
                       ku: c_int, alpha: *const c_float_complex, a: *const c_float_complex,
                       lda: c_int, x: *const c_float_complex, incx: c_int,
                       beta: *const c_float_complex, y: *mut c_float_complex, incy: c_int);
    pub fn cblas_ctrmv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, a: *const c_float_complex, lda: c_int,
                       x: *mut c_float_complex, incx: c_int);
    pub fn cblas_ctbmv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, k: c_int, a: *const c_float_complex, lda: c_int,
                       x: *mut c_float_complex, incx: c_int);
    pub fn cblas_ctpmv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, ap: *const c_float_complex,
                       x: *mut c_float_complex, incx: c_int);
    pub fn cblas_ctrsv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, a: *const c_float_complex, lda: c_int,
                       x: *mut c_float_complex, incx: c_int);
    pub fn cblas_ctbsv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, k: c_int, a: *const c_float_complex, lda: c_int,
                       x: *mut c_float_complex, incx: c_int);
    pub fn cblas_ctpsv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, ap: *const c_float_complex,
                       x: *mut c_float_complex, incx: c_int);

    pub fn cblas_zgemv(layout: CblasLayout, transa: CblasTranspose, m: c_int, n: c_int,
                       alpha: *const c_double_complex, a: *const c_double_complex, lda: c_int,
                       x: *const c_double_complex, incx: c_int, beta: *const c_double_complex,
                       y: *mut c_double_complex, incy: c_int);
    pub fn cblas_zgbmv(layout: CblasLayout, transa: CblasTranspose, m: c_int, n: c_int, kl: c_int,
                       ku: c_int, alpha: *const c_double_complex, a: *const c_double_complex,
                       lda: c_int, x: *const c_double_complex, incx: c_int,
                       beta: *const c_double_complex, y: *mut c_double_complex, incy: c_int);
    pub fn cblas_ztrmv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, a: *const c_double_complex, lda: c_int,
                       x: *mut c_double_complex, incx: c_int);
    pub fn cblas_ztbmv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, k: c_int, a: *const c_double_complex, lda: c_int,
                       x: *mut c_double_complex, incx: c_int);
    pub fn cblas_ztpmv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, ap: *const c_double_complex,
                       x: *mut c_double_complex, incx: c_int);
    pub fn cblas_ztrsv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, a: *const c_double_complex, lda: c_int,
                       x: *mut c_double_complex, incx: c_int);
    pub fn cblas_ztbsv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, k: c_int, a: *const c_double_complex, lda: c_int,
                       x: *mut c_double_complex, incx: c_int);
    pub fn cblas_ztpsv(layout: CblasLayout, uplo: CblasUplo, transa: CblasTranspose,
                       diag: CblasDiag, n: c_int, ap: *const c_double_complex,
                       x: *mut c_double_complex, incx: c_int);

    // Prefixes S and D only
    pub fn cblas_ssymv(layout: CblasLayout, uplo: CblasUplo, n: c_int, alpha: c_float,
                       a: *const c_float, lda: c_int, x: *const c_float, incx: c_int,
                       beta: c_float, y: *mut c_float, incy: c_int);
    pub fn cblas_ssbmv(layout: CblasLayout, uplo: CblasUplo, n: c_int, k: c_int, alpha: c_float,
                       a: *const c_float, lda: c_int, x: *const c_float, incx: c_int,
                       beta: c_float, y: *mut c_float, incy: c_int);
    pub fn cblas_sspmv(layout: CblasLayout, uplo: CblasUplo, n: c_int, alpha: c_float,
                       ap: *const c_float, x: *const c_float, incx: c_int, beta: c_float,
                       y: *mut c_float, incy: c_int);
    pub fn cblas_sger(layout: CblasLayout, m: c_int, n: c_int, alpha: c_float, x: *const c_float,
                      incx: c_int, y: *const c_float, incy: c_int, a: *mut c_float, lda: c_int);
    pub fn cblas_ssyr(layout: CblasLayout, uplo: CblasUplo, n: c_int, alpha: c_float,
                      x: *const c_float, incx: c_int, a: *mut c_float, lda: c_int);
    pub fn cblas_sspr(layout: CblasLayout, uplo: CblasUplo, n: c_int, alpha: c_float,
                      x: *const c_float, incx: c_int, ap: *mut c_float);
    pub fn cblas_ssyr2(layout: CblasLayout, uplo: CblasUplo, n: c_int, alpha: c_float,
                       x: *const c_float, incx: c_int, y: *const c_float, incy: c_int,
                       a: *mut c_float, lda: c_int);
    pub fn cblas_sspr2(layout: CblasLayout, uplo: CblasUplo, n: c_int, alpha: c_float,
                       x: *const c_float, incx: c_int, y: *const c_float, incy: c_int,
                       a: *mut c_float);

    pub fn cblas_dsymv(layout: CblasLayout, uplo: CblasUplo, n: c_int, alpha: c_double,
                       a: *const c_double, lda: c_int, x: *const c_double, incx: c_int,
                       beta: c_double, y: *mut c_double, incy: c_int);
    pub fn cblas_dsbmv(layout: CblasLayout, uplo: CblasUplo, n: c_int, k: c_int, alpha: c_double,
                       a: *const c_double, lda: c_int, x: *const c_double, incx: c_int,
                       beta: c_double, y: *mut c_double, incy: c_int);
    pub fn cblas_dspmv(layout: CblasLayout, uplo: CblasUplo, n: c_int, alpha: c_double,
                       ap: *const c_double, x: *const c_double, incx: c_int, beta: c_double,
                       y: *mut c_double, incy: c_int);
    pub fn cblas_dger(layout: CblasLayout, m: c_int, n: c_int, alpha: c_double, x: *const c_double,
                      incx: c_int, y: *const c_double, incy: c_int, a: *mut c_double, lda: c_int);
    pub fn cblas_dsyr(layout: CblasLayout, uplo: CblasUplo, n: c_int, alpha: c_double,
                      x: *const c_double, incx: c_int, a: *mut c_double, lda: c_int);
    pub fn cblas_dspr(layout: CblasLayout, uplo: CblasUplo, n: c_int, alpha: c_double,
                      x: *const c_double, incx: c_int, ap: *mut c_double);
    pub fn cblas_dsyr2(layout: CblasLayout, uplo: CblasUplo, n: c_int, alpha: c_double,
                       x: *const c_double, incx: c_int, y: *const c_double, incy: c_int,
                       a: *mut c_double, lda: c_int);
    pub fn cblas_dspr2(layout: CblasLayout, uplo: CblasUplo, n: c_int, alpha: c_double,
                       x: *const c_double, incx: c_int, y: *const c_double, incy: c_int,
                       a: *mut c_double);

    // Prefixes C and Z only
    pub fn cblas_chemv(layout: CblasLayout, uplo: CblasUplo, n: c_int,
                       alpha: *const c_float_complex, a: *const c_float_complex, lda: c_int,
                       x: *const c_float_complex, incx: c_int, beta: *const c_float_complex,
                       y: *mut c_float_complex, incy: c_int);
    pub fn cblas_chbmv(layout: CblasLayout, uplo: CblasUplo, n: c_int, k: c_int,
                       alpha: *const c_float_complex, a: *const c_float_complex, lda: c_int,
                       x: *const c_float_complex, incx: c_int, beta: *const c_float_complex,
                       y: *mut c_float_complex, incy: c_int);
    pub fn cblas_chpmv(layout: CblasLayout, uplo: CblasUplo, n: c_int,
                       alpha: *const c_float_complex, ap: *const c_float_complex,
                       x: *const c_float_complex, incx: c_int, beta: *const c_float_complex,
                       y: *mut c_float_complex, incy: c_int);
    pub fn cblas_cgeru(layout: CblasLayout, m: c_int, n: c_int, alpha: *const c_float_complex,
                       x: *const c_float_complex, incx: c_int, y: *const c_float_complex,
                       incy: c_int, a: *mut c_float_complex, lda: c_int);
    pub fn cblas_cgerc(layout: CblasLayout, m: c_int, n: c_int, alpha: *const c_float_complex,
                       x: *const c_float_complex, incx: c_int, y: *const c_float_complex,
                       incy: c_int, a: *mut c_float_complex, lda: c_int);
    pub fn cblas_cher(layout: CblasLayout, uplo: CblasUplo, n: c_int, alpha: c_float,
                      x: *const c_float_complex, incx: c_int, a: *mut c_float_complex, lda: c_int);
    pub fn cblas_chpr(layout: CblasLayout, uplo: CblasUplo, n: c_int, alpha: c_float,
                      x: *const c_float_complex, incx: c_int, a: *mut c_float_complex);
    pub fn cblas_cher2(layout: CblasLayout, uplo: CblasUplo, n: c_int,
                       alpha: *const c_float_complex, x: *const c_float_complex, incx: c_int,
                       y: *const c_float_complex, incy: c_int, a: *mut c_float_complex,
                       lda: c_int);
    pub fn cblas_chpr2(layout: CblasLayout, uplo: CblasUplo, n: c_int,
                       alpha: *const c_float_complex, x: *const c_float_complex, incx: c_int,
                       y: *const c_float_complex, incy: c_int, ap: *mut c_float_complex);

    pub fn cblas_zhemv(layout: CblasLayout, uplo: CblasUplo, n: c_int,
                       alpha: *const c_double_complex, a: *const c_double_complex, lda: c_int,
                       x: *const c_double_complex, incx: c_int, beta: *const c_double_complex,
                       y: *mut c_double_complex, incy: c_int);
    pub fn cblas_zhbmv(layout: CblasLayout, uplo: CblasUplo, n: c_int, k: c_int,
                       alpha: *const c_double_complex, a: *const c_double_complex, lda: c_int,
                       x: *const c_double_complex, incx: c_int, beta: *const c_double_complex,
                       y: *mut c_double_complex, incy: c_int);
    pub fn cblas_zhpmv(layout: CblasLayout, uplo: CblasUplo, n: c_int,
                       alpha: *const c_double_complex, ap: *const c_double_complex,
                       x: *const c_double_complex, incx: c_int, beta: *const c_double_complex,
                       y: *mut c_double_complex, incy: c_int);
    pub fn cblas_zgeru(layout: CblasLayout, m: c_int, n: c_int, alpha: *const c_double_complex,
                       x: *const c_double_complex, incx: c_int, y: *const c_double_complex,
                       incy: c_int, a: *mut c_double_complex, lda: c_int);
    pub fn cblas_zgerc(layout: CblasLayout, m: c_int, n: c_int, alpha: *const c_double_complex,
                       x: *const c_double_complex, incx: c_int, y: *const c_double_complex,
                       incy: c_int, a: *mut c_double_complex, lda: c_int);
    pub fn cblas_zher(layout: CblasLayout, uplo: CblasUplo, n: c_int, alpha: c_double,
                      x: *const c_double_complex, incx: c_int, a: *mut c_double_complex,
                      lda: c_int);
    pub fn cblas_zhpr(layout: CblasLayout, uplo: CblasUplo, n: c_int, alpha: c_double,
                      x: *const c_double_complex, incx: c_int, a: *mut c_double_complex);
    pub fn cblas_zher2(layout: CblasLayout, uplo: CblasUplo, n: c_int,
                       alpha: *const c_double_complex, x: *const c_double_complex, incx: c_int,
                       y: *const c_double_complex, incy: c_int, a: *mut c_double_complex,
                       lda: c_int);
    pub fn cblas_zhpr2(layout: CblasLayout, uplo: CblasUplo, n: c_int,
                       alpha: *const c_double_complex, x: *const c_double_complex, incx: c_int,
                       y: *const c_double_complex, incy: c_int, ap: *mut c_double_complex);
"""

level3 = """
    // Standard prefixes (S, D, C, and Z)
    pub fn cblas_sgemm(layout: CblasLayout, transa: CblasTranspose, transb: CblasTranspose,
                       m: c_int, n: c_int, k: c_int, alpha: c_float, a: *const c_float, lda: c_int,
                       b: *const c_float, ldb: c_int, beta: c_float, c: *mut c_float, ldc: c_int);
    pub fn cblas_ssymm(layout: CblasLayout, side: CblasSide, uplo: CblasUplo, m: c_int, n: c_int,
                       alpha: c_float, a: *const c_float, lda: c_int, b: *const c_float,
                       ldb: c_int, beta: c_float, c: *mut c_float, ldc: c_int);
    pub fn cblas_ssyrk(layout: CblasLayout, uplo: CblasUplo, trans: CblasTranspose, n: c_int,
                       k: c_int, alpha: c_float, a: *const c_float, lda: c_int, beta: c_float,
                       c: *mut c_float, ldc: c_int);
    pub fn cblas_ssyr2k(layout: CblasLayout, uplo: CblasUplo, trans: CblasTranspose, n: c_int,
                        k: c_int, alpha: c_float, a: *const c_float, lda: c_int, b: *const c_float,
                        ldb: c_int, beta: c_float, c: *mut c_float, ldc: c_int);
    pub fn cblas_strmm(layout: CblasLayout, side: CblasSide, uplo: CblasUplo,
                       transa: CblasTranspose, diag: CblasDiag, m: c_int, n: c_int, alpha: c_float,
                       a: *const c_float, lda: c_int, b: *mut c_float, ldb: c_int);
    pub fn cblas_strsm(layout: CblasLayout, side: CblasSide, uplo: CblasUplo,
                       transa: CblasTranspose, diag: CblasDiag, m: c_int, n: c_int, alpha: c_float,
                       a: *const c_float, lda: c_int, b: *mut c_float, ldb: c_int);

    pub fn cblas_dgemm(layout: CblasLayout, transa: CblasTranspose, transb: CblasTranspose,
                       m: c_int, n: c_int, k: c_int, alpha: c_double, a: *const c_double,
                       lda: c_int, b: *const c_double, ldb: c_int, beta: c_double,
                       c: *mut c_double, ldc: c_int);
    pub fn cblas_dsymm(layout: CblasLayout, side: CblasSide, uplo: CblasUplo, m: c_int, n: c_int,
                       alpha: c_double, a: *const c_double, lda: c_int, b: *const c_double,
                       ldb: c_int, beta: c_double, c: *mut c_double, ldc: c_int);
    pub fn cblas_dsyrk(layout: CblasLayout, uplo: CblasUplo, trans: CblasTranspose, n: c_int,
                       k: c_int, alpha: c_double, a: *const c_double, lda: c_int, beta: c_double,
                       c: *mut c_double, ldc: c_int);
    pub fn cblas_dsyr2k(layout: CblasLayout, uplo: CblasUplo, trans: CblasTranspose, n: c_int,
                        k: c_int, alpha: c_double, a: *const c_double, lda: c_int,
                        b: *const c_double, ldb: c_int, beta: c_double, c: *mut c_double,
                        ldc: c_int);
    pub fn cblas_dtrmm(layout: CblasLayout, side: CblasSide, uplo: CblasUplo,
                       transa: CblasTranspose, diag: CblasDiag, m: c_int, n: c_int,
                       alpha: c_double, a: *const c_double, lda: c_int, b: *mut c_double,
                       ldb: c_int);
    pub fn cblas_dtrsm(layout: CblasLayout, side: CblasSide, uplo: CblasUplo,
                       transa: CblasTranspose, diag: CblasDiag, m: c_int, n: c_int,
                       alpha: c_double, a: *const c_double, lda: c_int, b: *mut c_double,
                       ldb: c_int);

    pub fn cblas_cgemm(layout: CblasLayout, transa: CblasTranspose, transb: CblasTranspose,
                       m: c_int, n: c_int, k: c_int, alpha: *const c_float_complex,
                       a: *const c_float_complex, lda: c_int, b: *const c_float_complex,
                       ldb: c_int, beta: *const c_float_complex, c: *mut c_float_complex,
                       ldc: c_int);
    pub fn cblas_csymm(layout: CblasLayout, side: CblasSide, uplo: CblasUplo, m: c_int, n: c_int,
                       alpha: *const c_float_complex, a: *const c_float_complex, lda: c_int,
                       b: *const c_float_complex, ldb: c_int, beta: *const c_float_complex,
                       c: *mut c_float_complex, ldc: c_int);
    pub fn cblas_csyrk(layout: CblasLayout, uplo: CblasUplo, trans: CblasTranspose, n: c_int,
                       k: c_int, alpha: *const c_float_complex, a: *const c_float_complex,
                       lda: c_int, beta: *const c_float_complex, c: *mut c_float_complex,
                       ldc: c_int);
    pub fn cblas_csyr2k(layout: CblasLayout, uplo: CblasUplo, trans: CblasTranspose, n: c_int,
                        k: c_int, alpha: *const c_float_complex, a: *const c_float_complex,
                        lda: c_int, b: *const c_float_complex, ldb: c_int,
                        beta: *const c_float_complex, c: *mut c_float_complex, ldc: c_int);
    pub fn cblas_ctrmm(layout: CblasLayout, side: CblasSide, uplo: CblasUplo,
                       transa: CblasTranspose, diag: CblasDiag, m: c_int, n: c_int,
                       alpha: *const c_float_complex, a: *const c_float_complex, lda: c_int,
                       b: *mut c_float_complex, ldb: c_int);
    pub fn cblas_ctrsm(layout: CblasLayout, side: CblasSide, uplo: CblasUplo,
                       transa: CblasTranspose, diag: CblasDiag, m: c_int, n: c_int,
                       alpha: *const c_float_complex, a: *const c_float_complex, lda: c_int,
                       b: *mut c_float_complex, ldb: c_int);

    pub fn cblas_zgemm(layout: CblasLayout, transa: CblasTranspose, transb: CblasTranspose,
                       m: c_int, n: c_int, k: c_int, alpha: *const c_double_complex,
                       a: *const c_double_complex, lda: c_int, b: *const c_double_complex,
                       ldb: c_int, beta: *const c_double_complex, c: *mut c_double_complex,
                       ldc: c_int);
    pub fn cblas_zsymm(layout: CblasLayout, side: CblasSide, uplo: CblasUplo, m: c_int, n: c_int,
                       alpha: *const c_double_complex, a: *const c_double_complex, lda: c_int,
                       b: *const c_double_complex, ldb: c_int, beta: *const c_double_complex,
                       c: *mut c_double_complex, ldc: c_int);
    pub fn cblas_zsyrk(layout: CblasLayout, uplo: CblasUplo, trans: CblasTranspose, n: c_int,
                       k: c_int, alpha: *const c_double_complex, a: *const c_double_complex,
                       lda: c_int, beta: *const c_double_complex, c: *mut c_double_complex,
                       ldc: c_int);
    pub fn cblas_zsyr2k(layout: CblasLayout, uplo: CblasUplo, trans: CblasTranspose, n: c_int,
                        k: c_int, alpha: *const c_double_complex, a: *const c_double_complex,
                        lda: c_int, b: *const c_double_complex, ldb: c_int,
                        beta: *const c_double_complex, c: *mut c_double_complex, ldc: c_int);
    pub fn cblas_ztrmm(layout: CblasLayout, side: CblasSide, uplo: CblasUplo,
                       transa: CblasTranspose, diag: CblasDiag, m: c_int, n: c_int,
                       alpha: *const c_double_complex, a: *const c_double_complex, lda: c_int,
                       b: *mut c_double_complex, ldb: c_int);
    pub fn cblas_ztrsm(layout: CblasLayout, side: CblasSide, uplo: CblasUplo,
                       transa: CblasTranspose, diag: CblasDiag, m: c_int, n: c_int,
                       alpha: *const c_double_complex, a: *const c_double_complex, lda: c_int,
                       b: *mut c_double_complex, ldb: c_int);

    // Prefixes C and Z only
    pub fn cblas_chemm(layout: CblasLayout, side: CblasSide, uplo: CblasUplo, m: c_int, n: c_int,
                       alpha: *const c_float_complex, a: *const c_float_complex, lda: c_int,
                       b: *const c_float_complex, ldb: c_int, beta: *const c_float_complex,
                       c: *mut c_float_complex, ldc: c_int);
    pub fn cblas_cherk(layout: CblasLayout, uplo: CblasUplo, trans: CblasTranspose, n: c_int,
                       k: c_int, alpha: c_float, a: *const c_float_complex, lda: c_int,
                       beta: c_float, c: *mut c_float_complex, ldc: c_int);
    pub fn cblas_cher2k(layout: CblasLayout, uplo: CblasUplo, trans: CblasTranspose, n: c_int,
                        k: c_int, alpha: *const c_float_complex, a: *const c_float_complex,
                        lda: c_int, b: *const c_float_complex, ldb: c_int, beta: c_float,
                        c: *mut c_float_complex, ldc: c_int);

    pub fn cblas_zhemm(layout: CblasLayout, side: CblasSide, uplo: CblasUplo, m: c_int, n: c_int,
                       alpha: *const c_double_complex, a: *const c_double_complex, lda: c_int,
                       b: *const c_double_complex, ldb: c_int, beta: *const c_double_complex,
                       c: *mut c_double_complex, ldc: c_int);
    pub fn cblas_zherk(layout: CblasLayout, uplo: CblasUplo, trans: CblasTranspose, n: c_int,
                       k: c_int, alpha: c_double, a: *const c_double_complex, lda: c_int,
                       beta: c_double, c: *mut c_double_complex, ldc: c_int);
    pub fn cblas_zher2k(layout: CblasLayout, uplo: CblasUplo, trans: CblasTranspose, n: c_int,
                        k: c_int, alpha: *const c_double_complex, a: *const c_double_complex,
                        lda: c_int, b: *const c_double_complex, ldb: c_int, beta: c_double,
                        c: *mut c_double_complex, ldc: c_int);
"""

name_re = re.compile("\s*pub fn cblas_(\w+)")
argument_re = re.compile("(\w+): ([^,]*)(,|\))")
return_re = re.compile("(?:\s*->\s*([^;]+))?");

def pull_name(s):
    match = name_re.match(s)
    assert(match is not None)
    return match.group(1), s[match.end(1):]

def pull_argument(s):
    match = argument_re.match(s)
    if match is None:
        return None, None, s

    return match.group(1), match.group(2), s[match.end(3):]

def pull_return(s):
    match = return_re.match(s)
    if match is None:
        return None, s

    return match.group(1), s[match.end(1):]

def chew(s, c):
    assert(s[0] == c)
    return s[1:]

class Func(object):
    def __init__(self, level, name, args, ret):
        self.level = level
        self.name = name
        self.args = args
        self.ret = ret

    @staticmethod
    def parse(level, line):
        name, line = pull_name(line)
        if name is None:
            return None

        line = chew(line, '(')
        args = []
        while True:
            arg, aty, line = pull_argument(line)
            if arg is None:
                break
            args.append((arg, aty))
            line = line.strip()

        ret, line = pull_return(line)

        return Func(level, name, args, ret)

def translate_argument(name, cty, f):
    if cty == "c_int":
        return "usize"

    elif cty == "CblasDiag":
        return "Diagonal"
    elif cty == "CblasLayout":
        return "Layout"
    elif cty == "CblasSide":
        return "Side"
    elif cty == "CblasTranspose":
        return "Transpose"
    elif cty == "CblasUplo":
        return "Triangular"

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
    if rty == "usize":
        return "{} as c_int".format(name)

    elif rty in ["Diagonal", "Layout", "Side", "Transpose", "Triangular"]:
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
    if cty == "CblasIndex":
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
    s.append("ffi::cblas_{}(".format(f.name))

    indent = 8 + 11 + len(f.name) + 1
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
    return [Func.parse(level, line) for line in lines]

def do(funcs):
    for f in funcs:
        print("#[inline]")
        print(format_header(f))
        print(format_body(f))
        print("}\n")

do(prepare(1, level1_functions))
do(prepare(1, level1_routines))
do(prepare(2, level2))
do(prepare(3, level3))
