#!/usr/bin/env python

from common import Function, read_functions
from documentation import print_documentation
import argparse
import os
import re

level_scalars = {
    1: ["alpha", "a", "b", "c", "z"],
    2: ["alpha", "beta"],
    3: ["alpha", "beta"],
}

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sys', required=True)
    parser.add_argument('--doc')
    arguments = parser.parse_args()
    sections = read_functions(os.path.join(arguments.sys, 'src', 'c.rs'))
    assert(len(sections) == 5)
    do(prepare(1, sections[0]), arguments.doc)
    do(prepare(1, sections[1]), arguments.doc)
    do(prepare(2, sections[2]), arguments.doc)
    do(prepare(3, sections[3]), arguments.doc)
