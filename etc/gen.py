# little script to mostly-generate src/metal.rs from some massaged function
# signatures. Probably not useful anymore now that it has been done once!

import re

all_sigs = """
rotg(a: *mut c_float, b: *mut c_float, c: *mut c_float, s: *mut c_float);
scal(n: *const c_int, a: *const c_float, x: *mut c_float, incx: *const c_int);
axpy(n: *const c_int, alpha: *const c_float, x: *const c_float, incx: *const c_int, y: *mut c_float, incy: *const c_int);
nrm2(n: *const c_int, x: *const c_float, incx: *const c_int) -> c_float;
asum(n: *const c_int, x: *const c_float, incx: *const c_int) -> c_float;
gemv(trans: *const c_char, m: *const c_int, n: *const c_int, alpha: *const c_float, a: *const c_float, lda: *const c_int, x: *const c_float, incx: *const c_int, beta: *const c_float, y: *mut c_float, incy: *const c_int);
gbmv(trans: *const c_char, m: *const c_int, n: *const c_int, kl: *const c_int, ku: *const c_int, alpha: *const c_float, a: *const c_float, lda: *const c_int, x: *const c_float, incx: *const c_int, beta: *const c_float, y: *mut c_float, incy: *const c_int);
trmv(uplo: *const c_char, transa: *const c_char, diag: *const c_char, n: *const c_int, a: *const c_float, lda: *const c_int, b: *mut c_float, incx: *const c_int);
tbmv(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int, k: *const c_int, a: *const c_float, lda: *const c_int, x: *mut c_float, incx: *const c_int);
tpmv(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int, ap: *const c_float, x: *mut c_float, incx: *const c_int);
trsv(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int, a: *const c_float, lda: *const c_int, x: *mut c_float, incx: *const c_int);
tbsv(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int, k: *const c_int, a: *const c_float, lda: *const c_int, x: *mut c_float, incx: *const c_int);
tpsv(uplo: *const c_char, trans: *const c_char, diag: *const c_char, n: *const c_int, ap: *const c_float, x: *mut c_float, incx: *const c_int);
gemm(transa: *const c_char, transb: *const c_char, m: *const c_int, n: *const c_int, k: *const c_int, alpha: *const c_float, a: *const c_float, lda: *const c_int, b: *const c_float, ldb: *const c_int, beta: *const c_float, c: *mut c_float, ldc: *const c_int);
symm(side: *const c_char, uplo: *const c_char, m: *const c_int, n: *const c_int, alpha: *const c_float, a: *const c_float, lda: *const c_int, b: *const c_float, ldb: *const c_int, beta: *const c_float, c: *mut c_float, ldc: *const c_int);
syrk(uplo: *const c_char, trans: *const c_char, n: *const c_int, k: *const c_int, alpha: *const c_float, a: *const c_float, lda: *const c_int, beta: *const c_float, c: *mut c_float, ldc: *const c_int);
syr2k(uplo: *const c_char, trans: *const c_char, n: *const c_int, k: *const c_int, alpha: *const c_float, a: *const c_float, lda: *const c_int, b: *const c_float, ldb: *const c_int, beta: *const c_float, c: *mut c_float, ldc: *const c_int);
trmm(side: *const c_char, uplo: *const c_char, transa: *const c_char, diag: *const c_char, m: *const c_int, n: *const c_int, alpha: *const c_float, a: *const c_float, lda: *const c_int, b: *mut c_float, ldb: *const c_int);
trsm(side: *const c_char, uplo: *const c_char, transa: *const c_char, diag: *const c_char, m: *const c_int, n: *const c_int, alpha: *const c_float, a: *const c_float, lda: *const c_int, b: *mut c_float, ldb: *const c_int);
"""

real_sigs = """
dot(n: *const c_int, x: *const c_float, incx: *const c_int, y: *const c_float, incy: *const c_int) -> c_float;
dsdot(n: *const c_int, x: *const c_float, incx: *const c_int, y: *const c_float, incy: *const c_int) -> c_float;
rot(n: *const c_int, x: *mut c_float, incx: *const c_int, y: *mut c_float, incy: *const c_int, c: *const c_float, s: *const c_float);
rotm(n: *const c_int, x: *mut c_float, incx: *const c_int, y: *mut c_float, incy: *const c_int, param: *const c_float);
rotmg(d1: *mut c_float, d2: *mut c_float, x1: *mut c_float, y1: *const c_float, param: *mut c_float);
syr(uplo: *const c_char, n: *const c_int, alpha: *const c_float, x: *const c_float, incx: *const c_int, a: *mut c_float, lda: *const c_int);
spr(uplo: *const c_char, n: *const c_int, alpha: *const c_float, x: *const c_float, incx: *const c_int, ap: *mut c_float);
syr2(uplo: *const c_char, n: *const c_int, alpha: *const c_float, x: *const c_float, incx: *const c_int, y: *const c_float, incy: *const c_int, a: *mut c_float, lda: *const c_int);
spr2(uplo: *const c_char, n: *const c_int, alpha: *const c_float, x: *const c_float, incx: *const c_int, y: *const c_float, incy: *const c_int, ap: *mut c_float);
symv(uplo: *const c_char, n: *const c_int, alpha: *const c_float, a: *const c_float, lda: *const c_int, x: *const c_float, incx: *const c_int, beta: *const c_float, y: *mut c_float, incy: *const c_int);
sbmv(uplo: *const c_char, n: *const c_int, k: *const c_int, alpha: *const c_float, a: *const c_float, lda: *const c_int, x: *const c_float, incx: *const c_int, beta: *const c_float, y: *mut c_float, incy: *const c_int);
spmv(uplo: *const c_char, n: *const c_int, alpha: *const c_float, ap: *const c_float, x: *const c_float, incx: *const c_int, beta: *const c_float, y: *mut c_float, incy: *const c_int);
ger(m: *const c_int, n: *const c_int, alpha: *const c_float, x: *const c_float, incx: *const c_int, y: *const c_float, incy: *const c_int, a: *mut c_float, lda: *const c_int);
"""

complex_sigs = """
dotu(pres: *mut complex_double, n: *const c_int, x: *const complex_double, incx: *const c_int, y: *const complex_double, incy: *const c_int);
dotc(pres: *mut complex_double, n: *const c_int, x: *const complex_double, incx: *const c_int, y: *const complex_double, incy: *const c_int);
hemv(uplo: *const c_char, n: *const c_int, alpha: *const complex_float, a: *const complex_float, lda: *const c_int, x: *const complex_float, incx: *const c_int, beta: *const complex_float, y: *mut complex_float, incy: *const c_int);
hbmv(uplo: *const c_char, n: *const c_int, k: *const c_int, alpha: *const complex_float, a: *const complex_float, lda: *const c_int, x: *const complex_float, incx: *const c_int, beta: *const complex_float, y: *mut complex_float, incy: *const c_int);
hpmv(uplo: *const c_char, n: *const c_int, alpha: *const complex_float, ap: *const complex_float, x: *const complex_float, incx: *const c_int, beta: *const complex_float, y: *mut complex_float, incy: *const c_int);
geru(m: *const c_int, n: *const c_int, alpha: *const complex_float, x: *const complex_float, incx: *const c_int, y: *const complex_float, incy: *const c_int, a: *mut complex_float, lda: *const c_int);
gerc(m: *const c_int, n: *const c_int, alpha: *const complex_float, x: *const complex_float, incx: *const c_int, y: *const complex_float, incy: *const c_int, a: *mut complex_float, lda: *const c_int);
her(uplo: *const c_char, n: *const c_int, alpha: *const c_float, x: *const complex_float, incx: *const c_int, a: *mut complex_float, lda: *const c_int);
hpr(uplo: *const c_char, n: *const c_int, alpha: *const c_float, x: *const complex_float, incx: *const c_int, ap: *mut complex_float);
hpr2(uplo: *const c_char, n: *const c_int, alpha: *const complex_float, x: *const complex_float, incx: *const c_int, y: *const complex_float, incy: *const c_int, ap: *mut complex_float);
her2(uplo: *const c_char, n: *const c_int, alpha: *const complex_float, x: *const complex_float, incx: *const c_int, y: *const complex_float, incy: *const c_int, a: *mut complex_float, lda: *const c_int);
hemm(side: *const c_char, uplo: *const c_char, m: *const c_int, n: *const c_int, alpha: *const complex_float, a: *const complex_float, lda: *const c_int, b: *const complex_float, ldb: *const c_int, beta: *const complex_float, c: *mut complex_float, ldc: *const c_int);
herk(uplo: *const c_char, trans: *const c_char, n: *const c_int, k: *const c_int, alpha: *const c_float, a: *const complex_float, lda: *const c_int, beta: *const c_float, c: *mut complex_float, ldc: *const c_int);
her2k(uplo: *const c_char, trans: *const c_char, n: *const c_int, k: *const c_int, alpha: *const complex_float, a: *const complex_float, lda: *const c_int, b: *const complex_float, ldb: *const c_int, beta: *const c_float, c: *mut complex_float, ldc: *const c_int);
"""

id_re = re.compile("(\w+)")
arg_re = re.compile("(\w+): ([^,]*)(,|\))")
ret_re = re.compile("(?:\s*->\s*([^;]+))?;");

def pull_ident(s):
    m = id_re.match(s)
    if m is None:
        return None, s
    return m.group(1), s[m.end(1):]

def pull_arg(s):
    m = arg_re.match(s)
    if m is None:
        return None, None, s

    return m.group(1), m.group(2), s[m.end(3):]

def pull_ret(s):
    m = ret_re.match(s)
    if m is None:
        return None, s

    return m.group(1), s[m.end(1):]

def chew(s, c):
    assert s[0] == c
    return s[1:]

class Func(object):
    def __init__(self, name, args, ret):
        self.name = name
        self.args = args
        self.ret = ret

    def __str__(self):
        return "{}{} -> {}".format(self.name, self.args, self.ret)

    @staticmethod
    def parse(line):
        name, line = pull_ident(line)
        if name is None:
            return None
        line = chew(line, '(')
        args = []
        while True:
            arg, ty, line = pull_arg(line)
            if arg is None:
                break
            args.append((arg, ty))
            line = line.strip()

        ret, line = pull_ret(line)
        return Func(name, args, ret)

def translate_type(name, ty, fty):
    if name == "uplo":
        return "Uplo"
    elif name.startswith("trans"):
        return "Trans"
    elif name == "side":
        return "Side"
    elif name == "diag":
        return "Diag"
    elif name.startswith("ld") or name.startswith("inc"):
        return "usize"
    elif name == "alpha" or name == "beta":
        return fty
    elif ty == "*const c_float":
        return "&[{}]".format(fty)
    elif ty == "*mut c_float":
        return "&mut [{}]".format(fty)
    elif ty == "*const complex_float" or ty == "*const complex_double":
        return "&[{}]".format(fty)
    elif ty == "*mut complex_float" or ty == "*mut complex_double":
        return "&mut [{}]".format(fty)
    elif name == "m" or name == "n" or name == "k" or name == "kl" or name == "ku":
        return "usize"
    else:
        print ("idgaf about {}: {}".format(name, ty))
        return "idgaf"

def format_body(name, fty, args):
    s = []
    s.append("unsafe {\n")
    s.append("raw::{}_(".format(name))
    for arg in args:
        name, ty = arg
        realty = translate_type(name, ty, fty)
        if name == "uplo" or name == "diag" or name == "side" or name.startswith("trans"):
            s.append("&({} as c_char) as *const _,\n".format(name))
        elif realty == "usize":
            s.append("&({} as int) as *const _,\n".format(name))
        elif realty == "Complex<f32>" or realty == "Complex<f64>" or realty == "f32" or realty == "f64":
            s.append("&{} as *const _ as *const _,\n".format(name))
        elif realty.startswith("&mut ["):
            s.append("{}.as_mut_ptr() as *mut _,\n".format(name))
        elif realty.startswith("&["):
            s.append("{}.as_ptr() as *const _,\n".format(name))
        else:
            s.append("unreachable!(idgaf about {}),\n".format(name))

        #ty = translate_type(*arg, fty=fty)
    s.append(")\n")
    s.append("}")
    return "".join(s)

def format_args(l, fty):
    s = []
    for arg in l:
        s.append(arg[0])
        s.append(": ")
        s.append(translate_type(*arg, fty=fty))
        s.append(", ")
    return "".join(s)

all_funcs = [Func.parse(line) for line in all_sigs.split('\n') if line]
real_funcs = [Func.parse(line) for line in real_sigs.split('\n') if line]
complex_funcs = [Func.parse(line) for line in complex_sigs.split('\n') if line]

def do_normal(fty, fpr):
    for f in all_funcs:
        print("#[inline]")
        print("pub fn {}{}({}){} {{".format(fpr, f.name, format_args(f.args, fty), "" if f.ret is None else " -> {}".format(f.ret)))
        print(format_body("{}{}".format(fpr, f.name), fty, f.args))
        print("}")

def do_real(fty, fpr):
    for f in real_funcs:
        print("#[inline]")
        print("pub fn {}{}({}){} {{".format(fpr, f.name, format_args(f.args, fty), "" if f.ret is None else " -> {}".format(f.ret)))
        print(format_body("{}{}".format(fpr, f.name), fty, f.args))
        print("}")

def do_complex(fty, fpr):
    for f in complex_funcs:
        print("#[inline]")
        print("pub fn {}{}({}){} {{".format(fpr, f.name, format_args(f.args, fty), "" if f.ret is None else " -> {}".format(f.ret)))
        print(format_body("{}{}".format(fpr, f.name), fty, f.args))
        print("}")

do_normal("f32", "s")
do_normal("f64", "d")
do_normal("Complex<f32>", "c")
do_normal("Complex<f64>", "z")

do_real("f32", "s")
do_real("f64", "d")

do_complex("Complex<f32>", "c")
do_complex("Complex<f64>", "z")
