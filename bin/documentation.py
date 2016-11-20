import os, re, sys

def clean(lines):
    lines = [re.sub(r"^\*> ?", "", line) for line in lines]

    while len(lines) > 0 and lines[0].strip() == "":
        lines = lines[1:]

    while len(lines) > 0 and lines[-1].strip() == "":
        lines = lines[:-1]

    if all(line.startswith("   ") or line.strip() == "" for line in lines):
        lines = [re.sub(r"^   ", "", line) for line in lines]

    return lines

def extract(filename):
    SEARCHING = 0
    IN_PURPOSE = 1
    IN_DESCRIPTION = 2

    lines = []
    state = SEARCHING
    with open(filename) as file:
        for line in file:
            if state == SEARCHING:
                if "par Purpose" in line:
                    state = IN_PURPOSE
            elif state == IN_PURPOSE:
                if "\\verbatim" in line:
                    state = IN_DESCRIPTION
            elif state == IN_DESCRIPTION:
                if "\\endverbatim" in line:
                    break
                lines.append(line.replace("\n", ""))

    return lines

def flush(line):
    print("/// {}".format(line).strip())

def print_documentation(f, reference):
    filename = os.path.join(reference, "BLAS", "SRC", "{}.f".format(f.name))
    if not os.path.exists(filename):
        return

    lines = clean(extract(filename))
    if len(lines) == 0:
        return

    first = re.sub(r"(?i)\s*{}\s+".format(f.name), "", lines[0])
    flush(first.strip().capitalize())
    for line in lines[1:]:
        if line.startswith("   "):
            flush("```text")
            flush(line[3:])
            flush("```")
        else:
            flush(line)
