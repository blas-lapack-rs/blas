import os, re, sys

class Blob:
    def append(self, _):
        pass

    def finish(self, _, __, ___):
        pass

class Formula(Blob):
    def __init__(self):
        self.lines = []

    def append(self, line):
        self.lines.append(line)

    def format(self, output):
        output.write("/// ```text\n")
        for line in self.lines:
            output.write("/// {}\n".format(line[3:]))
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
        if index == 0:
            first = re.sub(r"(?i)\s*{}\s+".format(f.name), "", self.lines[0])
            first = first.strip().capitalize()
            self.lines[0] = first

        if index == total - 1 and self.lines[-1][-1] != '.':
            self.lines[-1] = "{}.".format(self.lines[-1])

        text = re.sub(r"\s+", " ", " ".join(self.lines))
        lines = text.split(". ")
        lines[1:] = [line.capitalize() for line in lines[1:]]
        text = ". ".join(lines)

        chunks = text.split(" ")
        lines = []
        count = 0
        while len(chunks) > 0:
            current = " ".join(chunks[:count])
            if count == len(chunks) or 4 + len(current) + len(chunks[0]) > 80:
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
