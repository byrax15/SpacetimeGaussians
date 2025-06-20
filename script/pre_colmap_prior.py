#!python
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import itertools
import shutil
import subprocess
import time
from typing import Literal, NamedTuple, Optional, Tuple
from tap import Tap, tapify
from pathlib import Path
import re
import os


def average(self):
    return (self.min + self.max) / 2


def standard_deviation(self):
    return (self.max - self.min) / 4


class RandomPointsArgs(Tap):
    """Skip COLMAP feature extraction and matching, and use a provided sparse model for all frames."""
    mode = "share_full_prior"  # DO NOT USE, reserved for subparser disambiguation
    # if >0, overrides points3D.txt with points sampled from a normal distribution, else use the file as-is
    gen_random_points: int = 0
    min: float = -10.
    max: float = 10.
    color: Tuple[int, ...] = (200, 200, 200)


class ArgumentParser(Tap):
    mode = "default"  # DO NOT USE, reserved for subparser disambiguation
    videosdir: Path
    prior_data_fmt: str = "txt"
    imageext: str = "png"
    no_single_camera: bool = False
    camera_model: str = "SIMPLE_PINHOLE"
    mapper_ba_tolerance: float = 1e-6
    dryrun: bool = False
    parallel: bool = False
    startframe: int = 0
    endframe: int

    def configure(self):
        self.add_subparser("share_full_prior", RandomPointsArgs)


class SparsePriorModel:
    def __init__(self, videosdir: Path, prior_data_fmt: str):
        self.prior = videosdir / "prior"
        self.cameras = self.prior / f"cameras.{prior_data_fmt}"
        self.images = self.prior / f"images.{prior_data_fmt}"
        self.points3D = self.prior / f"points3D.{prior_data_fmt}"

        assert self.prior.exists()
        assert self.cameras.exists()
        assert self.images.exists()
        assert self.points3D.exists()

    def __repr__(self):
        return f"""SparseModel(
prior = {self.prior},
cameras = {self.cameras},
images = {self.images},
points3D = {self.points3D}
)"""


def colmap_input_dir(point_dir: Path, frame: int) -> Path:
    return point_dir / f"colmap_{frame}" / "input"


def colmap_input_image_path(point_dir: Path, frame: int, cam_num: int, imageext: str) -> Path:
    return colmap_input_dir(point_dir, frame) / f"cam{cam_num:03}.{imageext}"


args = ArgumentParser().parse_args()
prior_data = SparsePriorModel(args.videosdir, args.prior_data_fmt)
frames_dir = args.videosdir / "frames"
assert frames_dir.exists()
point_dir = args.videosdir / "point"

for f in range(args.startframe, args.endframe):
    colmap_input_dir(point_dir, f).mkdir(parents=True, exist_ok=True)

cam_pattern = re.compile(r"cam(\d+)")
frame_pattern = re.compile(r"(\d+)")
for cam in frames_dir.glob("cam*"):
    cam_num = int(cam_pattern.match(cam.name)[1])
    for frame in cam.glob(f"*.{args.imageext}"):
        frame_num = int(frame_pattern.match(frame.name)[1])
        link_path = colmap_input_image_path(
            point_dir, frame_num, cam_num, args.imageext)
        if args.dryrun:
            print(f'ln -s -r {frame} {link_path}')
        else:
            link_path.symlink_to(os.path.relpath(
                frame, link_path.parent), False)

if args.mode == "share_full_prior":
    args: RandomPointsArgs = args
    if args.gen_random_points > 0:
        import numpy
        # Generate random points using a normal distribution
        with (prior_data.prior/"points3D.txt").open('w') as points3d:
            r, g, b = args.color
            for i in range(args.gen_random_points):
                x, y, z = numpy.random.normal(
                    average(args), standard_deviation(args), 3)
                points3d.write(
                    f"{i} {x} {y} {z} {r} {g} {b} 0 0 0 1 1 2 2\n")
    for bin in itertools.chain(prior_data.prior.glob("*.bin"), prior_data.prior.glob("*.ply")):
        bin.unlink()
    os.system(
        f"colmap model_converter --output_type BIN --output_path {prior_data.prior} --input_path {prior_data.prior}")
    for dest in point_dir.glob("colmap_*"):
        try:
            (dest/"sparse").mkdir(parents=True, exist_ok=True)
            (dest/"sparse"/"0").symlink_to(
                os.path.relpath(prior_data.prior, dest/"sparse/0/.."), True)
            (dest/"images").symlink_to(
                os.path.relpath(dest/"input", dest/'images/..'), True)
        except FileExistsError:
            pass
    exit(0)


@dataclass
class UndistortedSparsePath:
    src: Path
    dst: Path


class ColmapExecutor:
    def __init__(self, point_dir: Path, frame_num: int, no_single_camera: bool, camera_model: str, dryrun: bool, mapper_ba_tolerance: float):
        self.frame_dir = point_dir / f"colmap_{frame_num}"
        self.undistorted_sparse_path = self.frame_dir / "sparse" / "0"
        self.database_path = self.frame_dir / "distorted" / "database.db"
        self.input_path = self.frame_dir / "input"
        self.sparse_path = self.frame_dir / "distorted" / "sparse" / "0"
        self.dryrun = dryrun
        self.commands = [
            f"""colmap feature_extractor
--database_path {self.database_path}
--image_path {self.input_path}
--ImageReader.single_camera {0 if no_single_camera else 1}
--ImageReader.camera_model {camera_model}
            """,
            f"""colmap exhaustive_matcher
--database_path {self.database_path}
            """,
            f"""colmap point_triangulator
--database_path {self.database_path}
--image_path {self.input_path}
--output_path {self.sparse_path}
--input_path {prior_data.prior}
--Mapper.ba_global_function_tolerance={mapper_ba_tolerance}
            """,
            f"""colmap image_undistorter
--image_path {self.input_path}
--input_path {self.sparse_path}
--output_path {self.frame_dir}
--output_type COLMAP
            """,
            f"""mkdir -p {self.frame_dir/'sparse/0'}""",
            f"""mv {self.frame_dir/'sparse/*'} {self.frame_dir/'sparse/0/'}/"""
        ]

    def __call__(self):
        for i, command in enumerate(self.commands):
            if (self.dryrun):
                print(command)
            else:
                for dir in [self.sparse_path]:
                    dir.mkdir(parents=True, exist_ok=True)
                code = os.system(command.replace('\n', ' '))
                if code != 0:
                    raise Exception(
                        f"Command failed with code {code}: {command}")


tasks = [ColmapExecutor(point_dir, f, args.no_single_camera, args.camera_model, args.dryrun, args.mapper_ba_tolerance)
         for f in range(args.startframe, args.endframe)]
print("Starting COLMAP Processing...")
start_time = time.monotonic()
if args.parallel:
    # run every task in tasks in a separate process
    with ProcessPoolExecutor() as pool:
        asyncs = [pool.submit(t.__call__) for t in tasks]
        [a.result() for a in asyncs]
        pool.shutdown(wait=True)
else:
    [t() for t in tasks]  # [] forces evaluation, () is lazy
end_time = time.monotonic()
print("Duration (s): ",  end_time - start_time)
