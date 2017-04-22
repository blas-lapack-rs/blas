import os, re, sys

class Blob:
    def append(self, *_):
        pass

    def finish(self, *_):
        pass

class Formula(Blob):
    def __init__(self):
        self.lines = []

    def append(self, line):
        self.lines.append(line)

    def finish(self, *_):
        for i, line in enumerate(self.lines):
            assert(len(line[:3].strip()) == 0)
            line = re.sub(r"\s+", " ", line[3:])
            line = re.sub(r"\(\s+", "(", line)
            line = re.sub(r"\s+\)", ")", line)
            self.lines[i] = line

    def format(self, output):
        output.write("/// ```text\n")
        for line in self.lines:
            output.write("/// {}\n".format(line))
        output.write("/// ```\n")

class Space(Blob):
    def format(self, output):
        output.write("///\n")

class Text(Blob):
    def __init__(self):
        self.lines = []

    def append(self, line):
        self.lines.append(line)

    def finish(self, index, total, f):
        lines = self.lines

        if index == 0:
            first = re.sub(r"(?i)\s*{}\s+".format(f.name), "", lines[0])
            lines[0] = first

        line = " ".join(lines)

        # Heuristic: We expect that if the text contains an equal sign (`=`) then
        # it is not safe to reformat the text.
        #
        # Holds for:
        # * https://github.com/Reference-LAPACK/lapack/blob/master/BLAS/SRC/drotm.f
        # * https://github.com/Reference-LAPACK/lapack/blob/master/BLAS/SRC/drotmg.f
        # * https://github.com/Reference-LAPACK/lapack/blob/master/BLAS/SRC/dsdot.f
        # * https://github.com/Reference-LAPACK/lapack/blob/master/BLAS/SRC/srotm.f
        # * https://github.com/Reference-LAPACK/lapack/blob/master/BLAS/SRC/srotmg.f
        #
        if "=" in line:
            pass
        else:
            line = re.sub(r"\s+", " ", line)
            line = re.sub(r"\(\s+", "(", line)
            line = re.sub(r"\s+\)", ")", line)

            if index == total - 1 and line[-1] != ".":
                line = "{}.".format(line)

            lines = line.split(". ")
            lowercase = ["alpha", "or", "where"]
            for i, line in enumerate(lines):
                if all([not line.startswith(word) for word in lowercase]):
                    lines[i] = "{}{}".format(line[0].upper(), line[1:])
            line = ". ".join(lines)

            substitutes = {
                "Compute": "Computes",
                "equal to 1": "equal to one",
            }
            for key, value in substitutes.items():
                line = re.sub(r"\b{}\b".format(key), value, line)

            chunks = line.split(" ")
            lines = []
            count = 0
            while len(chunks) > 0:
                current = " ".join(chunks[:count])
                if count == len(chunks) or 4 + len(current) + len(chunks[count]) >= 80:
                    lines.append(current)
                    chunks = chunks[count:]
                    count = 0
                else:
                    count += 1

        self.lines = lines

    def format(self, output):
        for line in self.lines:
            output.write("/// {}\n".format(line))

def clean(lines):
    lines = [re.sub(r"^\*> ?", "", line.rstrip()) for line in lines]

    while len(lines) > 0 and lines[0].strip() == "":
        lines = lines[1:]

    while len(lines) > 0 and lines[-1].strip() == "":
        lines = lines[:-1]

    margin = 42
    for line in lines:
        if len(line.strip()) > 0:
            margin = min(margin, len(line) - len(line.strip()))
    for i, line in enumerate(lines):
        if len(line.strip()) > 0:
            lines[i] = lines[i][margin:]

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

def partition(lines):
    paragraphs = []
    current = None

    for line in lines:
        if line.startswith("   "):
            klass = Formula
        elif len(line) == 0:
            klass = Space
        else:
            klass = Text
        if not isinstance(current, klass):
            if current is not None:
                paragraphs.append(current)
            current = klass()
        current.append(line)

    if current is not None:
        paragraphs.append(current)

    return paragraphs

def print_documentation(f, reference):
    filename = os.path.join(reference, "BLAS", "SRC", "{}.f".format(f.name))
    if not os.path.exists(filename):
        return
    paragraphs = partition(clean(extract(filename)))
    for i, paragraph in enumerate(paragraphs):
        paragraph.finish(i, len(paragraphs), f)
        paragraph.format(sys.stdout)
