#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import argparse
import os
from pathlib import Path
from typing import Optional
import torch
from random import randint
import random
import sys
import uuid
import time
import json

import numpy as np
import cv2
from tqdm import tqdm
import shutil

sys.path.append("./thirdparty/gaussian_splatting")
from thirdparty.gaussian_splatting.utils.general_utils import safe_state  # NOQA
from argparse import ArgumentParser, Namespace  # NOQA
from thirdparty.gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args  # NOQA


def getparser():
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    # we put more parameters in optimization params, just for convenience.
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6029)
    parser.add_argument('--debug_from', type=int, default=-2)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+",
                        type=int, default=[7_000, 10000, 12000, 25_000, 30_000])
    parser.add_argument("--test_iterations", default=-1, type=int)

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations",
                        nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--densify", type=int, default=1,
                        help="densify =1, we control points on N3d dataset")
    parser.add_argument("--duration", type=int, default=5,
                        help="5 debug , 50 used")
    parser.add_argument("--basicfunction", type=str, default="gaussian")
    parser.add_argument("--rgbfunction", type=str, default="rgbv1")
    parser.add_argument("--rdpip", type=str, default="v2")
    parser.add_argument("--configpath", type=str, default="None")
    parser.add_argument("--yield-loss", "--yield_loss", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # incase we provide config file not directly pass to the file
    if os.path.exists(args.configpath) and args.configpath != "None":
        print("overload config from " + args.configpath)
        config = json.load(open(args.configpath))
        for k in config.keys():
            try:
                value = getattr(args, k)
                newvalue = config[k]
                setattr(args, k, newvalue)
            except:
                print("failed set config: " + k)
        print("finish load config from " + args.configpath)
    else:
        raise ValueError("config file not exist or not provided")

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    return args, lp.extract(args), op.extract(args), pp.extract(args)


def getrenderparts(render_pkg):
    return render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]


def gettestparse():
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--test_iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--multiview", action="store_true",
                        help="Skip metric evaluation. Useful for multiview datasets.")
    parser.add_argument("--duration", default=50, type=int)
    parser.add_argument("--rgbfunction", type=str, default="rgbv1")
    parser.add_argument("--rdpip", type=str, default="v3")
    parser.add_argument("--valloader", type=str, default="colmap")
    parser.add_argument("--configpath", type=str, default="1")
    parser.add_argument("--cameras_validate_all", action="store_true",
                        help="Render all cameras in the colmap model, not only the training cameras.")
    parser.add_argument("-oc", "--override_config",
                        nargs='*', type=lambda s: s.split("="), default=[],
                        help="Override config parameters specified in the config file. Format: -oc key1=value1 key2=value2 ...")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    # configpath
    safe_state(args.quiet)

    args.multiview |= args.valloader.endswith("mv")

    if os.path.exists(args.configpath) and args.configpath != "None":
        print("overload config from " + args.configpath)
        config = json.load(open(args.configpath))
        for k in config.keys():
            try:
                value = getattr(args, k)
                newvalue = config[k]
                setattr(args, k, newvalue)
            except:
                print("failed set config: " + k)
        print("finish load config from " + args.configpath)

    if hasattr(parser, "_actions"):
        def type_func(k, v):  # type: ignore
            typef = next(
                (a.type for a in parser._actions if a.dest == k), None)
            try:
                return typef(v)  # type: ignore
            except:
                return v
    else:
        def type_func(k, v): return v

    for k, v in args.override_config:
        setattr(args, k, type_func(k, v))

    print("args: " + str(args))

    return args, model.extract(args), pipeline.extract(args), args.multiview


def getcolmapsinglen3d(folder, offset, single_camera=False):

    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = f"colmap feature_extractor --database_path {dbfile} --image_path {inputimagefolder} --ImageReader.single_camera {1 if single_camera else 0}"

    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)

    featurematcher = "colmap exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)

   # threshold is from   https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/scripts/local_colmap_and_resize.sh#L62
    triandmap = "colmap point_triangulator --database_path " + dbfile + " --image_path " + inputimagefolder + " --output_path " + distortedmodel \
        + " --input_path " + manualinputfolder + \
        " --Mapper.ba_global_function_tolerance=0.000001"

    exit_code = os.system(triandmap)
    if exit_code != 0:
        exit(exit_code)
    print(triandmap)

    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel + " --output_path " + folder  \
        + " --output_type COLMAP"
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    removeinput = "rm -r " + inputimagefolder
    exit_code = os.system(removeinput)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)


def getcolmapsingleimundistort(folder, offset):

    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor SiftExtraction.max_image_size 6000 --database_path " + \
        dbfile + " --image_path " + inputimagefolder

    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)

    featurematcher = "colmap exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)

    triandmap = "colmap point_triangulator --database_path " + dbfile + " --image_path " + inputimagefolder + " --output_path " + distortedmodel \
        + " --input_path " + manualinputfolder + \
        " --Mapper.ba_global_function_tolerance=0.000001"

    exit_code = os.system(triandmap)
    if exit_code != 0:
        exit(exit_code)
    print(triandmap)

    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel + " --output_path " + folder  \
        + " --output_type COLMAP "  # --blank_pixels 1
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    removeinput = "rm -r " + inputimagefolder
    exit_code = os.system(removeinput)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    # Copy each file from the source directory to the destination directory
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)


def getcolmapsingleimdistort(folder, offset):

    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor SiftExtraction.max_image_size 6000 --database_path " + \
        dbfile + " --image_path " + inputimagefolder

    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)

    featurematcher = "colmap exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)

    triandmap = "colmap point_triangulator --database_path " + dbfile + " --image_path " + inputimagefolder + " --output_path " + distortedmodel \
        + " --input_path " + manualinputfolder + \
        " --Mapper.ba_global_function_tolerance=0.000001"

    exit_code = os.system(triandmap)
    if exit_code != 0:
        exit(exit_code)
    print(triandmap)

    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel + " --output_path " + folder  \
        + " --output_type COLMAP "  # --blank_pixels 1
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    removeinput = "rm -r " + inputimagefolder
    exit_code = os.system(removeinput)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)


def getcolmapsingletechni(folder, offset):

    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor --database_path " + \
        dbfile + " --image_path " + inputimagefolder

    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)

    featurematcher = "colmap exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)

    triandmap = "colmap point_triangulator --database_path " + dbfile + " --image_path " + inputimagefolder + " --output_path " + distortedmodel \
        + " --input_path " + manualinputfolder + \
        " --Mapper.ba_global_function_tolerance=0.000001"

    exit_code = os.system(triandmap)
    if exit_code != 0:
        exit(exit_code)
    print(triandmap)

    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel + " --output_path " + folder  \
        + " --output_type COLMAP "  #
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)

    return
