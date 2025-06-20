#!/bin/python3
import os
import argparse
import sys
import re
import shutil
import concurrent.futures
import pathlib

range_re = re.compile('^range[(](?:(?:(?:\\d+)(?:,\\s?)?){1,3})[)]$')


def to_range(sarg: str) -> range:
    if range_re.match(sarg) is None:
        raise argparse.ArgumentTypeError(
            "Argument must be a 'range(start,end,step)' expression.")
    else:
        return eval(sarg)


def to_inclusive_range(sarg: str) -> range:
    r = to_range(sarg)
    return range(r.start, r.stop+1, r.step)


class Executor:
    @staticmethod
    def dry_run(src: pathlib.Path, dst: pathlib.Path):
        pass

    @staticmethod
    def run(src: pathlib.Path, dst: pathlib.Path):
        pass


class Copyier(Executor):
    @staticmethod
    def dry_run(src: pathlib.Path, dst: pathlib.Path):
        print(f"mkdir -p {os.path.dirname(dst)}")
        print(f"cp {src} {dst}")

    @staticmethod
    def run(src: pathlib.Path, dst: pathlib.Path):
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"Error copying {src} to {dst}: {e}")


class Symlinker(Executor):
    @staticmethod
    def dry_run(src: pathlib.Path, dst: pathlib.Path):
        print(f"mkdir -p {os.path.dirname(dst)}")
        print(f"ln -s {{tgt:{src}}} {{link:{dst}}}")

    @staticmethod
    def run(src: pathlib.Path, dst: pathlib.Path):
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            dst.symlink_to(src)
        except Exception as e:
            print(f"Error linking {src} to {dst}: {e}")
            raise


parser = argparse.ArgumentParser(prog="cp_dataset")
parser.add_argument("-n", "--dry_run", action="store_true",
                    help="Print the commands that would be executed.")
parser.add_argument("-f", "--frame_range", required=True,
                    type=to_inclusive_range, help="An inclusive range(start,end,step).")
parser.add_argument("-c", "--camera_range", required=True,
                    type=to_inclusive_range, help="An inclusive range(start,end,step).")
parser.add_argument("-b", "--bases", default=[0, 1], type=int, nargs=2, help="The base index for the output's frame and camera.")
parser.add_argument("-p", "--parallel", action="store_true",
                    help="Execute the copy/link operations in parallel.")
executorgroup = parser.add_mutually_exclusive_group()
executorgroup.add_argument('-ec', '--exec_copy', action="store_true")
executorgroup.add_argument("-el", '--exec_link', action='store_true')
executorgroup.set_defaults(exec_copy=True, exec_link=False)
parser.add_argument("-s", "--srcfmt", required=True, type=str,
                    help="Format string describing the copied source files, must contain 'frame' and 'cam' format variables.")
parser.add_argument("-d", "--dstfmt", required=True, type=str,
                    help="Format string describing the copied source files, must contain 'frame' and 'cam' format variables.")
args = parser.parse_args()


with concurrent.futures.ThreadPoolExecutor() as executor:
    executor_type = Symlinker if args.exec_link else Copyier
    func = executor_type.dry_run if args.dry_run else executor_type.run
    dispatch = (lambda src, dst: executor.submit(
        func, src, dst)) if args.parallel else func

    dst_frame = args.bases[0]
    for frame in args.frame_range:
        dst_cam = args.bases[1]
        for cam in args.camera_range:
            src = args.srcfmt.format(frame=frame, cam=cam)
            src = os.path.expanduser(src)
            src = os.path.expandvars(src)

            dst = args.dstfmt.format(frame=dst_frame, cam=dst_cam)
            dst = os.path.expanduser(dst)
            dst = os.path.expandvars(dst)

            dispatch(pathlib.Path(src), pathlib.Path(dst))
            # dispatch(pathlib.Path(src),pathlib.Path(dst))
            # executor.submit(dispatch,pathlib.Path(src),pathlib.Path(dst))
            dst_cam += 1
        dst_frame += 1
