# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import re

class RoutineDesc(object):
    def __init__(self, name, short_doc, level):
        self.name = name
        self.short_doc = short_doc
        self.level = level

ROUTINES = [
    RoutineDesc("rotg", "setup Givens rotation", 1),
    RoutineDesc("rotmg", "setup modified Givens rotation", 1),
    RoutineDesc("rot", "apply Givens rotation", 1),
    RoutineDesc("rotm", "apply modified Givens rotation", 1),
    RoutineDesc("swap", "swap x and y", 1),
    RoutineDesc("scal", u"x = α·x", 1),
    RoutineDesc("sscal", u"x = α·x, scalar α", 1),
    RoutineDesc("dscal", u"x = α·x, scalar α", 1),

    RoutineDesc("copy", "copy x into y", 1),
    RoutineDesc("axpy", u"y = α·x + y", 1),
    RoutineDesc("dot", "dot product", 1),
    RoutineDesc("dotu", u"y = x<sup>T</sup> · y, dot product", 1),
    RoutineDesc("dotc", u"y = x<sup>T</sup> · y, dot product with first argument conjugated", 1),
    RoutineDesc("dotu_sub", "dotu subroutine with return value as argument", 1),
    RoutineDesc("dotc_sub", "dotc subroutine with return value as argument", 1),
    RoutineDesc("sdot", "dot product with extended precision accumulation", 1),
    RoutineDesc("dsdot", "dot product with extended precision accumulation", 1),
    RoutineDesc("nrm2", "Euclidean norm", 1),
    RoutineDesc("asum", "sum of absolute values", 1),
    RoutineDesc("amax", "index of max abs value", 1),
    RoutineDesc("cabs1", "absolute value of complex number", 1),

    RoutineDesc("gemv", "matrix-vector multiply", 2),
    RoutineDesc("gbmv", "banded matrix-vector multiply", 2),
    RoutineDesc("hemv", "hermitian matrix-vector multiply", 2),
    RoutineDesc("hbmv", "hermitian banded matrix-vector multiply", 2),
    RoutineDesc("hpmv", "hermitian packed matrix-vector multiply", 2),
    RoutineDesc("symv", "symmetric matrix-vector multiply", 2),
    RoutineDesc("sbmv", "symmetric banded matrix-vector multiply", 2),
    RoutineDesc("spmv", "symmetric packed matrix-vector multiply", 2),
    RoutineDesc("trmv", "triangular matrix-vector multiply", 2),
    RoutineDesc("tbmv", "triangular banded matrix-vector multiply", 2),
    RoutineDesc("tpmv", "triangular packed matrix-vector multiply", 2),
    RoutineDesc("trsv", "solving triangular matrix problems", 2),
    RoutineDesc("tbsv", "solving triangular banded matrix problems", 2),
    RoutineDesc("tpsv", "solving triangular packed matrix problems", 2),
    RoutineDesc("ger", "performs the rank 1 operation A := α·x·y' + A", 2),
    RoutineDesc("geru", u"performs the rank 1 operation A := α·x·y' + A", 2),
    RoutineDesc("gerc", u"performs the rank 1 operation A := α·x·conjg( y' ) + A", 2),
    RoutineDesc("her", u"hermitian rank 1 operation A := α·x·conjg(x') + A", 2),
    RoutineDesc("hpr", u"hermitian packed rank 1 operation A := α·x·conjg( x' ) + A", 2),
    RoutineDesc("her2", "hermitian rank 2 operation", 2),
    RoutineDesc("hpr2", "hermitian packed rank 2 operation", 2),
    RoutineDesc("syr", u"performs the symmetric rank 1 operation A := α·x·x' + A", 2),
    RoutineDesc("spr", u"symmetric packed rank 1 operation A := α·x·x' + A", 2),
    RoutineDesc("syr2", u"performs the symmetric rank 2 operation, A := α·x·y' + α·y·x' + A", 2),
    RoutineDesc("spr2", u"performs the symmetric packed rank 2 operation, A := α·x·y' + α·y·x' + A", 2),

    RoutineDesc("gemm", "matrix-matrix multiply", 3),
    RoutineDesc("symm", "symmetric matrix-matrix multiply", 3),
    RoutineDesc("syrk", "symmetric rank-k update to a matrix", 3),
    RoutineDesc("syr2k", "symmetric rank-2k update to a matrix", 3),
    RoutineDesc("trmm", "triangular matrix-matrix multiply", 3),
    RoutineDesc("trsm", "solving triangular matrix with multiple right hand sides", 3),
    RoutineDesc("hemm", "hermitian matrix-matrix multiply", 3),
    RoutineDesc("herk", "hermitian rank-k update to a matrix", 3),
    RoutineDesc("her2k", "hermitian rank-2k update to a matrix", 3)
]

ROUTINE_DICT = {routine.name: routine for routine in ROUTINES}


class BlasFunction(object):
    """Parsed representation of a blas function."""
    LANGUAGE_C = "c"
    LANGUAGE_FORTRAN = "fortran"
    FUNC_NAME_RE = re.compile(r"(?P<lang>(cblas_)?)(?P<index>i?)(?P<argtype>[sdcz])(?P<routine>[^_][_a-z12]+)_?")
    TYPE_DESC = {"s": "real single-precision",
                 "d": "real double-precision",
                 "c": "complex single-precision",
                 "z": "complex double-precision",
                 "sc": "single-precision",
                 "dz": "double-precision",
                 "cs": "complex single-precision",
                 "zd": "complex double-precision"}

    def __init__(self, lang, argtype, returns_index, routine):
        self.lang = lang
        self.argtype = argtype
        self.returns_index = returns_index
        self.routine = routine

    @classmethod
    def parse(cls, func_name):
        """
        Given the name of a C or fortran BLAS function, return a parsed object representation.

        """
        m = cls.FUNC_NAME_RE.match(func_name)
        if m:
            if m.group("lang") == "cblas_":
                lang = cls.LANGUAGE_C
            else:
                lang = cls.LANGUAGE_FORTRAN
            argtype = m.group("argtype")
            returns_index = m.group("index") != ""
            routine = m.group("routine")

            # for some specific routines, we need to change the argtype
            if routine in ("cnrm2", "casum", "znrm2", "zasum", "srot", "drot"):
                argtype, routine = argtype + routine[0], routine[1:]

            return BlasFunction(lang, argtype, returns_index, routine)
        else:
            return None

    def format_name(self):
        """
        Render the name of the routine as it would appear in code.
        """
        lang_prefix, lang_suffix = {self.LANGUAGE_C: ("cblas_", ""), self.LANGUAGE_FORTRAN: ("", "_")}[self.lang]
        indexer = "i" if self.returns_index else ""
        return "{}{}{}{}{}".format(lang_prefix, indexer, self.argtype, self.routine, lang_suffix)

    def summary_desc(self):
        """
        Return the summary description of the routines.
        """
        if self.routine in ROUTINE_DICT:
            desc = ROUTINE_DICT[self.routine].short_doc
            return u"{} ({})".format(desc, self.TYPE_DESC[self.argtype])
        return ""


def format_documentation(f_name):
    bf = BlasFunction.parse(f_name)
    return "/// {}".format(bf.summary_desc()).encode("utf-8")

if __name__ == "__main__":
    f = BlasFunction.parse("cblas_sgemm")
    print("{}: {}".format(f.format_name(), f.summary_desc()))
