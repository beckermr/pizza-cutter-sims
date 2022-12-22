from ruamel.yaml import YAML
import glob

parser = YAML(typ="rt")
parser.indent(mapping=2, sequence=4, offset=2)
parser.width = 320
parser.default_flow_style = False

fnames = glob.glob("run*/config.yaml", recursive=True)

for fname in fnames:
    with open(fname, "r") as fp:
        lines = fp.readlines()
    new_lines = [
        ln for ln in lines if len(ln.strip()) > 0
    ]
    cfg = parser.load("".join(new_lines))

    if "residual_bkg" not in cfg["se"]:
        cfg["se"]["residual_bkg"] = 0.0
    if "residual_bkg_std" not in cfg["se"]:
        cfg["se"]["residual_bkg_std"] = 0.0

    with open(fname, "w") as fp:
        parser.dump(cfg, fp)

    with open(fname, "r") as fp:
        lines = fp.readlines()

    prev_indent = None
    new_lines = []
    for line in lines:
        curr_indent = len(line) - len(line.lstrip())
        if prev_indent is not None and prev_indent > curr_indent and curr_indent == 0:
            new_lines.append("\n")
        new_lines.append(line)
        prev_indent = curr_indent

    with open(fname, "w") as fp:
        fp.write("".join(new_lines))
