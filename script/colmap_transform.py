#!python

from __future__ import annotations
from collections import defaultdict
import contextlib
from dataclasses import dataclass, field
import os
import pathlib
import re
import sys
from typing import Generic, Literal, Tuple, TypeVar, Union
import weakref
import numpy as np
import pandas as pd
import scipy
import scipy.sparse
import scipy.spatial


@dataclass
class ColmapSparseModel:
    '''Parses a COLMAP sparse model as `pandas.DataFrame`s from the given base path.'''

    def __init__(self, path: pathlib.Path):
        self.cameras = self.load_cameras(path)
        self.images = self.load_images(path)
        self.points3d = self.load_points3d(path)

    @staticmethod
    def load_cameras(path: pathlib.Path) -> pd.DataFrame:
        rows = []
        p = path / 'cameras.txt'
        if not p.exists():
            return pd.DataFrame(rows)
        with p.open('r') as f:
            for line in f:
                if re.match(r'^\s*#', line):
                    continue
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                camera_id, model, width, height, *params = parts
                rows.append({
                    'camera_id': int(camera_id),
                    'model': model,
                    'width': int(width),
                    'height': int(height),
                    'params': [float(p) for p in params]
                })
        return pd.DataFrame(rows)

    @staticmethod
    def load_images(path: pathlib.Path) -> pd.DataFrame:
        rows = []
        p = path / 'images.txt'
        if not p.exists():
            return pd.DataFrame(rows)
        with p.open('r') as f:
            lines = [line for line in f if not re.match(r'^\s*#', line)]
        for i in range(0, len(lines), 2):
            image_line = lines[i].strip().split()
            if len(image_line) < 10:
                continue
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = image_line[:10]
            points2d_line = lines[i+1].strip().split()
            points2d = []
            for j in range(0, len(points2d_line), 3):
                x, y, point3D_id = points2d_line[j:j+3]
                points2d.append({
                    'x': float(x),
                    'y': float(y),
                    'point3D_id': int(point3D_id)
                })
            rows.append({
                'image_id': int(image_id),
                'qw': float(qw),
                'qx': float(qx),
                'qy': float(qy),
                'qz': float(qz),
                'tx': float(tx),
                'ty': float(ty),
                'tz': float(tz),
                'camera_id': int(camera_id),
                'name': name,
                'points2d': points2d
            })
        return pd.DataFrame(rows)

    @staticmethod
    def load_points3d(path: pathlib.Path) -> pd.DataFrame:
        rows = []
        p = path / 'points3D.txt'
        if not p.exists():
            return pd.DataFrame(rows)
        with p.open('r') as f:
            for line in f:
                if re.match(r'^\s*#', line):
                    continue
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                points3d_id, x, y, z, r, g, b, error, *track = parts
                track_list = []
                for i in range(0, len(track), 2):
                    if i+1 < len(track):
                        image_id, point2d_idx = track[i:i+2]
                        track_list.append({
                            'image_id': int(image_id),
                            'point2d_idx': int(point2d_idx)
                        })
                rows.append({
                    'points3d_id': int(points3d_id),
                    'x': float(x),
                    'y': float(y),
                    'z': float(z),
                    'r': int(r),
                    'g': int(g),
                    'b': int(b),
                    'error': float(error),
                    'track': track_list
                })
        return pd.DataFrame(rows)


class UnsupportedOptionError(Exception):
    """Raised when an option is not supported by a command."""

    def __init__(
            self, command: str, option_name, option_value, detail: str = '--{option_name} {option_value} is not supported.'):
        super().__init__(
            f'Command {command} Error: {detail.format(option_name=option_name, option_value=option_value)}')


@dataclass
class Print:
    """Pretty-print the files from the COLMAP sparse model."""
    file: list[Literal['cameras', 'images', 'points3d']] = \
        field(default_factory=lambda: ['cameras', 'images', 'points3d'])

    def call(self, model: ColmapSparseModel, model_path: pathlib.Path):
        for f in self.file:
            try:
                print(getattr(model, f))
            except:
                raise UnsupportedOptionError('print', 'file', self.file)


@dataclass
class Multiply:
    n_points: int
    colors:  list[tuple[int, int, int]]  # colors to use for the points
    color_weigths: list[float] = field(default_factory=list)
    '''weights for the colors, will default to equal weights if unspecified'''
    out: Literal['stdout', 'new', 'overwrite'] = 'stdout'
    '''either print to stdout, create a new file in the sparse model directory, or overwrite the existing one'''
    file: Literal['points3d'] = 'points3d'

    def call(self, model: ColmapSparseModel, model_path: pathlib.Path):
        if 'points3d' != self.file:
            raise UnsupportedOptionError('multiply', 'file', self.file)

        if self.color_weigths == []:
            self.color_weigths = [1 for _ in range(len(self.colors))]
        if len(self.color_weigths) != len(self.colors):
            raise UnsupportedOptionError(
                'multiply', 'color_weights', self.color_weigths,
                'if specified, color_weights must have the same length as colors')
        color_weight_sum = sum(self.color_weigths)
        for i, _ in enumerate(self.color_weigths):
            self.color_weigths[i] /= color_weight_sum

        @contextlib.contextmanager
        def get_file():
            if self.out == 'stdout':
                yield sys.stdout
            elif self.out == 'new':
                file = (model_path / 'points3D.new.txt').open('w')
                try:
                    yield file
                finally:
                    file.close()
            elif self.out == 'overwrite':
                file = (model_path / 'points3D.txt').open('w+')
                try:
                    yield file
                finally:
                    file.close()
            else:
                raise UnsupportedOptionError('multiply', 'out', self.out)

        # see for cam-to-world space definition: https://colmap.github.io/format.html#images-txt
        cam_qs = model.images[['qw', 'qx', 'qy', 'qz']]\
            .to_numpy()[:, [1, 2, 3, 0]]
        cam_ts = -model.images[['tx', 'ty', 'tz']].to_numpy()
        cams_world_pos = scipy.spatial.transform.Rotation \
            .from_quat(cam_qs) \
            .apply(cam_ts, inverse=True)

        # scale aabb centered on its origin
        min = cams_world_pos.min(0)
        max = cams_world_pos.max(0)
        dist = (max - min) * .5
        margined_dist = dist * .1
        min += margined_dist
        max -= margined_dist

        with get_file() as file:
            for i in range(self.n_points):
                point_pos = np.random.uniform(min, max)
                # randomly select a color using the weights
                color_idx = np.random.choice(
                    range(len(self.colors)), p=self.color_weigths)
                random_color = self.colors[color_idx]
                print(
                    '{0} {1} {2} {3} {4} {5} {6} 0 0 0 1 1 2 2'
                    .format(i, *point_pos, *random_color),
                    file=file)


if __name__ == "__main__":
    import tyro

    def main(sparse_model: pathlib.Path, call: Union[Print, Multiply]):
        call.call(ColmapSparseModel(sparse_model), sparse_model)

    tyro.cli(
        main,
        config=(tyro.conf.ConsolidateSubcommandArgs, tyro.conf.OmitSubcommandPrefixes))
