# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import re
import os

class RoutineDesc(object):
    def __init__(self, name, short_doc, level):
        self.name = name
        self.short_doc = short_doc
        self.level = level

LAPACK_ROOT = "../../ref-lapack"

def func_filename(func_name):
    """Return the filename that carries the implementation (and
description) of the given method."""
    return os.path.join(LAPACK_ROOT, "BLAS/SRC", "{}.f".format(func_name))

def purpose_section(func_name):
    """Return the purpose string, stripped of prefix characters, from the
file specified by filename.

    """
    filename = func_filename(func_name)

    if not os.path.exists(filename):
        return ""

    # Create a state machine that looks for a set
    SEARCHING = 0
    IN_PURPOSE = 1
    IN_DESC = 2

    parse_state = SEARCHING
    desc_lines = []

    with open(filename) as f:
        for line in f:
            if parse_state == SEARCHING:
                if "par Purpose" in line:
                    parse_state = IN_PURPOSE
            elif parse_state == IN_PURPOSE:
                if "\\verbatim" in line:
                    parse_state = IN_DESC
            elif parse_state == IN_DESC:
                if "\\endverbatim" in line:
                    break
                desc_lines.append(line)

    desc_lines = [re.sub(r"^\*> ?", "", line) for line in desc_lines]

    # Remove any starting and ending empty lines
    while len(desc_lines) > 0 and desc_lines[0].strip() == "":
        desc_lines = desc_lines[1:]
    while len(desc_lines) > 0 and desc_lines[-1].strip() == "":
        desc_lines = desc_lines[:-1]

    if len(desc_lines) == 0:
        return []

    # If all the lines are indented, unindent all of them
    if all(line.startswith("   ") or line.strip() == "" for line in desc_lines):
        desc_lines = [re.sub(r"^   ", "", line) for line in desc_lines]

    # Add an extra space to any blocks to make them code blocks, but only if it's not the first line.
    lines = [desc_lines[0]]
    for line in desc_lines[1:]:
        if line.startswith("   "):
            lines.extend(["```text\n", line, "```\n"])
        else:
            lines.append(line)
    # desc_lines = [re.sub(r"^   ([^ ])", r"    \1", line) for line in desc_lines]

    return lines

ROUTINES = [
    RoutineDesc("rotg", "setup Givens rotation", 1),
    RoutineDesc("rotmg", "setup modified Givens rotation", 1),
    RoutineDesc("rot", "apply Givens rotation", 1),
    RoutineDesc("rotm", "apply modified Givens rotation", 1),
    RoutineDesc("swap", "swap x and y", 1),
    RoutineDesc("scal", u"x = alpha*x", 1),
    RoutineDesc("sscal", u"x = alpha*x, scalar alpha", 1),
    RoutineDesc("dscal", u"x = alpha*x, scalar alpha", 1),

    RoutineDesc("copy", "copy x into y", 1),
    RoutineDesc("axpy", u"y = alpha*x + y", 1),
    RoutineDesc("dot", "dot product", 1),
    RoutineDesc("dotu", u"y = x<sup>T</sup> * y, dot product", 1),
    RoutineDesc("dotc", u"y = x<sup>T</sup> * y, dot product with first argument conjugated", 1),
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
    RoutineDesc("ger", "performs the rank 1 operation A := alpha*x*y' + A", 2),
    RoutineDesc("geru", u"performs the rank 1 operation A := alpha*x*y' + A", 2),
    RoutineDesc("gerc", u"performs the rank 1 operation A := alpha*x*conjg( y' ) + A", 2),
    RoutineDesc("her", u"hermitian rank 1 operation A := alpha*x*conjg(x') + A", 2),
    RoutineDesc("hpr", u"hermitian packed rank 1 operation A := alpha*x*conjg( x' ) + A", 2),
    RoutineDesc("her2", "hermitian rank 2 operation", 2),
    RoutineDesc("hpr2", "hermitian packed rank 2 operation", 2),
    RoutineDesc("syr", u"performs the symmetric rank 1 operation A := alpha*x*x' + A", 2),
    RoutineDesc("spr", u"symmetric packed rank 1 operation A := alpha*x*x' + A", 2),
    RoutineDesc("syr2", u"performs the symmetric rank 2 operation, A := alpha*x*y' + alpha*y*x' + A", 2),
    RoutineDesc("spr2", u"performs the symmetric packed rank 2 operation, A := alpha*x*y' + alpha*y*x' + A", 2),

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

    def canonical_name(self):
        """Return the canonical, language-independent, function_name."""
        indexer = "i" if self.returns_index else ""
        return "{}{}{}".format(indexer, self.argtype, self.routine)

    def format_name(self):
        """
        Render the name of the routine as it would appear in code.
        """
        lang_prefix, lang_suffix = {self.LANGUAGE_C: ("cblas_", ""), self.LANGUAGE_FORTRAN: ("", "_")}[self.lang]
        return "{}{}{}".format(lang_prefix, self.canonical_name(), lang_suffix)

    def short_desc(self):
        """
        Return the summary description of the routines.
        """
        if self.routine in ROUTINE_DICT:
            desc = ROUTINE_DICT[self.routine].short_doc
            return u"{} ({})".format(desc, self.TYPE_DESC[self.argtype])
        return ""

def format_documentation(f_name, f_args):
    bf = BlasFunction.parse(f_name)
    arg_names = {a[0] for a in f_args}

    # a, b, and c tend to be matrices, and thus capitalized.
    for c in "abc":
        if c in arg_names:
            arg_names = arg_names - set([c]) | set([c.upper()])

    doc_lines = purpose_section(bf.canonical_name())

    # replace variable names with encoded version, skipping code lines
    def variable_subst(line):
        if line.startswith("   "):
            return line
        for name in arg_names:
            line = re.sub(r"\b{}\b".format(name), r"`{}`".format(name), line)
        return line

    doc_lines = [variable_subst(line) for line in doc_lines]

    # comment this line to not include short descriptions
    doc_lines = [bf.short_desc().encode("utf-8") + "\n", "\n"] + doc_lines

    return "".join(["/// {}".format(line).encode("utf-8") for line in doc_lines]).rstrip()
