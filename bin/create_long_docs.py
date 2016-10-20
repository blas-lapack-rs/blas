from __future__ import print_function
import sys
import os
import re

def purpose_section(filename):
    """Return the purpose string, stripped of prefix characters, from the
file specified by filename.

    """
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
                desc_lines.append(line.replace("\n", ""))

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
            lines.extend(["```text", line, "```"])
        else:
            lines.append(line)
    # desc_lines = [re.sub(r"^   ([^ ])", r"    \1", line) for line in desc_lines]

    return lines

def _usage():
    print("""Generate the default documentation for blas functions.

usage: python {} NETLIB_REPOSITORY_ROOT.
""".format(sys.argv[1]))

if __name__ == "__main__":
    if len(sys.argv) < 2):
        usage()
        sys.exit(1)

    lapack_root = sys.argv[1]
    src_dir = os.path.join(lapack_root, "BLAS/SRC")

    print("ROUTINE_DOCS = {}")
    print("")

    for fn in os.listdir(src_dir):
        func_name, ext = os.path.splitext(fn)
        if ext != ".f":
            continue

        print('ROUTINE_DOCS["{}"] = """'.format(func_name))
        lines = purpose_section(os.path.join(src_dir, fn))
        print("\n".join('{}'.format(line) for line in lines))
        print('"""')
        print("")
