import os, re, sys

def extract(filename):
    SEARCHING = 0
    IN_PURPOSE = 1
    IN_DESCRIPTION = 2

    state = SEARCHING
    lines = []

    with open(filename) as f:
        for line in f:
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

    lines = [re.sub(r"^\*> ?", "", line) for line in lines]

    while len(lines) > 0 and lines[0].strip() == "":
        lines = lines[1:]
    while len(lines) > 0 and lines[-1].strip() == "":
        lines = lines[:-1]

    if len(lines) == 0:
        return []

    if all(line.startswith("   ") or line.strip() == "" for line in lines):
        lines = [re.sub(r"^   ", "", line) for line in lines]

    result = [lines[0]]
    for line in lines[1:]:
        if line.startswith("   "):
            result.extend(["```text", line, "```"])
        else:
            result.append(line)

    return result

def print_documentation(f, reference):
    filename = os.path.join(reference, "BLAS", "SRC", "{}.f".format(f.name))
    if not os.path.exists(filename):
        return
    lines = extract(filename)
    names = [arg[0] for arg in f.args]
    for line in lines:
        for name in names:
            line = re.sub(r"\b{}\b".format(name), r"`{}`".format(name), line)
        print("/// {}".format(line).strip())
