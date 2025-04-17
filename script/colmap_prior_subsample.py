from __future__ import annotations
from typing import Literal
from pathlib import Path
from tap import Tap
from pathlib import Path
import re
import shutil

NUMBER_CAMERAS = 1
NUMBER_HEADER_LINES = 1
NUMBER_LINES_PER_GROUP = 2


def immediate_invoke(f): return f()


OutModes = Literal["stdout", "newfile", "inplace"]


def out_lines(mode: OutModes, lines: list[str], inpath: Path):
    if mode == "stdout":
        print("".join(lines))
    elif mode == "newfile":
        outpath = inpath.with_name(inpath.name + ".new")
        with open(outpath, "w") as f:
            f.writelines(lines)
    elif mode == "inplace":
        outpath = Path(inpath)
        shutil.move(inpath, outpath.with_name(inpath.name + ".bak"))
        with open(outpath, "w") as f:
            f.writelines(lines)


class ArgumentParser(Tap):
    sparse_model_dir: Path
    camera_subsample: int = 1 # Preserve the first out of every N camera views
    camera_image_pattern = re.compile(r"([0-9]+)( .* )[0-9]+ Camera[.]([0-9]{3})\n$")
    """A pattern matching and grouping the camera id and image name; must contain 2 groups"""
    camera_new_name_format = "{image_id}{middle}0 cam{image_name:03}.png\n"
    camera_new_name_first_image = 1
    out = OutModes

    def configure(self):
        self.add_argument(
            "--out", choices=["stdout", "newfile", "inplace"], default="stdout")


args = ArgumentParser().parse_args()


@immediate_invoke
def to_single_camera():
    # convert to single camera
    cameras_path = args.sparse_model_dir / "cameras.txt"
    with open(cameras_path) as f:
        lines = f.readlines()

    filtered = lines[:1+NUMBER_HEADER_LINES]
    out_lines(args.out, filtered, cameras_path)


@immediate_invoke
def downsample_images():
    # downsample images file
    def select_line(line_num: int):
        if line_num < NUMBER_HEADER_LINES:
            return True
        return (line_num - NUMBER_HEADER_LINES) % (NUMBER_LINES_PER_GROUP * args.camera_subsample) < NUMBER_LINES_PER_GROUP

    image_path = args.sparse_model_dir / "images.txt"
    with open(image_path) as f:
        lines = f.readlines()

    filtered = [lines[i] for i in filter(select_line, range(len(lines)))]

    # Renumber images and relink to single camera
    image_number = 0
    for i, line in enumerate(filtered):
        image_name_match = args.camera_image_pattern.search(line)
        if image_name_match:
            filtered[i] = args.camera_new_name_format.format(
                image_id=image_number,
                middle=image_name_match.groups()[1],
                image_name=image_number+args.camera_new_name_first_image,
            )
            image_number += 1

    out_lines(args.out, filtered, image_path)
