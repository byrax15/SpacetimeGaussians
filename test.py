# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ========================================================================================================
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the thirdparty/gaussian_splatting/LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from collections import namedtuple
from dataclasses import dataclass
import json
from typing import Callable
import numpy as np
import time
from thirdparty.gaussian_splatting.arguments import ModelParams, PipelineParams
from skimage.metrics import structural_similarity as sk_ssim
from thirdparty.gaussian_splatting.helper3dg import gettestparse
from thirdparty.gaussian_splatting.scene.cameras import Camera
from thirdparty.gaussian_splatting.utils.image_utils import psnr
from thirdparty.gaussian_splatting.utils.loss_utils import ssim
from helper_train import getrenderpip, getmodel, trbfunction
from thirdparty.gaussian_splatting.lpipsPyTorch import lpips
import warnings
import scipy
import torchvision
from os import makedirs
from tqdm import tqdm
import os
from thirdparty.gaussian_splatting.scene import Scene
import torch
import sys
sys.path.append("./thirdparty/gaussian_splatting")


warnings.filterwarnings("ignore")

# modified from https://github.com/graphdeco-inria/gaussian-splatting/blob/main/render.py and https://github.com/graphdeco-inria/gaussian-splatting/blob/main/metrics.py


@dataclass
class PathPair:
    stem: str
    full: str


class SavePathTemplate:
    def __init__(self, duration, by_image_name_and_timestamp: bool):
        self.duration = duration
        self.__call__ = self._by_uid if not by_image_name_and_timestamp else self._by_image_name_and_timestamp

    def _by_image_name_and_timestamp(self, parent: str, view: Camera):
        stem = os.path.join(
            parent, f"{view.image_name}_{int(view.timestamp * self.duration):03d}.png")
        return PathPair(stem=stem, full=os.path.join(parent, stem))

    def _by_uid(self, parent: str, view: Camera):
        stem = f"{view.uid:05d}.png"
        return PathPair(stem=stem, full=os.path.join(parent, stem))

    def makedirs(self, parent: str, view: Camera):
        paths = self.__call__(parent, view)
        os.makedirs(os.path.dirname(paths.full), exist_ok=True)
        return paths


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, rbfbasefunction, rdpip, save_path_template: SavePathTemplate):
    render, GRsetting, GRzer = getrenderpip(rdpip)
    render_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if gaussians.rgbdecoder is not None:
        gaussians.rgbdecoder.cuda()
        gaussians.rgbdecoder.eval()
    statsdict = {}

    scales = gaussians.get_scaling

    scalemax = torch.amax(scales).item()
    scalesmean = torch.amin(scales).item()

    op = gaussians.get_opacity
    opmax = torch.amax(op).item()
    opmean = torch.mean(op).item()

    statsdict["scales_max"] = scalemax
    statsdict["scales_mean"] = scalesmean

    statsdict["op_max"] = opmax
    statsdict["op_mean"] = opmean

    statspath = os.path.join(model_path, "stat_" + str(iteration) + ".json")
    with open(statspath, 'w') as fp:
        json.dump(statsdict, fp, indent=True)

    psnrs = []
    lpipss = []
    lpipssvggs = []

    full_dict = {}
    per_view_dict = {}
    ssims = []
    ssimsv2 = []
    scene_dir = model_path
    image_names = []
    times = []

    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}

    full_dict[scene_dir][iteration] = {}
    per_view_dict[scene_dir][iteration] = {}

    if rdpip == "train_ours_full":
        # full model faster now when use fuse the rendering part (MLP) into cuda, same as 3dgs and instant-NGP.
        render, GRsetting, GRzer = getrenderpip("test_ours_full_fused")
    elif rdpip == "train_ours_lite":
        render, GRsetting, GRzer = getrenderpip("test_ours_lite")
    elif rdpip == "train_ours_fullss":
        render, GRsetting, GRzer = getrenderpip("test_ours_fullss_fused")
    elif rdpip == "train_ours_litess":
        render, GRsetting, GRzer = getrenderpip("test_ours_litess")
    else:
        render, GRsetting, GRzer = getrenderpip(rdpip)

    for idx, view in enumerate(tqdm(views, desc="Rendering and metric progress")):
        renderingpkg = render(view, gaussians, pipeline, background, scaling_modifier=1.0,
                              basicfunction=rbfbasefunction,  GRsetting=GRsetting, GRzer=GRzer)  # C x H x W
        rendering = renderingpkg["render"]
        rendering = torch.clamp(rendering, 0, 1.0)
        gt = view.original_image[0:3, :, :].cuda().float()
        ssims.append(ssim(rendering.unsqueeze(0), gt.unsqueeze(0)))

        psnrs.append(psnr(rendering.unsqueeze(0), gt.unsqueeze(0)))
        lpipss.append(lpips(rendering.unsqueeze(
            0), gt.unsqueeze(0), net_type='alex'))
        lpipssvggs.append(lpips(rendering.unsqueeze(
            0), gt.unsqueeze(0), net_type='vgg'))

        rendernumpy = rendering.permute(1, 2, 0).detach().cpu().numpy()
        gtnumpy = gt.permute(1, 2, 0).detach().cpu().numpy()

        ssimv2 = sk_ssim(rendernumpy, gtnumpy, multichannel=True,
                         channel_axis=2, data_range=1.)
        ssimsv2.append(ssimv2)

        render_paths = save_path_template.makedirs(render_path, view)
        gts_paths = save_path_template.makedirs(gts_path, view)
        torchvision.utils.save_image(rendering, render_paths.full)
        torchvision.utils.save_image(gt, gts_paths.full)
        image_names.append(render_paths.stem)

    for idx, view in enumerate(tqdm(views, desc="release gt images cuda memory for timing")):
        view.original_image = None  # .detach()
        torch.cuda.empty_cache()

    # start timing
    for _ in range(4):
        for idx, view in enumerate(tqdm(views, desc="timing ")):

            renderpack = render(view, gaussians, pipeline, background, scaling_modifier=1.0,
                                basicfunction=rbfbasefunction,  GRsetting=GRsetting, GRzer=GRzer)  # ["time"] # C x H x W
            duration = renderpack["duration"]
            if idx > 10:  # warm up
                times.append(duration)

    print(np.mean(np.array(times)))
    if len(views) > 0:
        full_dict[model_path][iteration].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                 "PSNR": torch.tensor(psnrs).mean().item(),
                                                 "LPIPS": torch.tensor(lpipss).mean().item(),
                                                 "ssimsv2": torch.tensor(ssimsv2).mean().item(),
                                                 "LPIPSVGG": torch.tensor(lpipssvggs).mean().item(),
                                                 "times": torch.tensor(times).mean().item()})

        per_view_dict[model_path][iteration].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                     "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                     "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                     "ssimsv2": {name: v for v, name in zip(torch.tensor(ssimsv2).tolist(), image_names)},
                                                     "LPIPSVGG": {name: lpipssvgg for lpipssvgg, name in zip(torch.tensor(lpipssvggs).tolist(), image_names)}, })

        with open(model_path + "/" + str(iteration) + "_runtimeresults.json", 'w') as fp:
            json.dump(full_dict, fp, indent=True)

        with open(model_path + "/" + str(iteration) + "_runtimeperview.json", 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)


# render free view
def render_setnogt(model_path, name, iteration, views, gaussians, pipeline, background, rbfbasefunction, rdpip, save_path_template: SavePathTemplate):
    render, GRsetting, GRzer = getrenderpip(rdpip)
    render_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "renders")

    makedirs(render_path, exist_ok=True)
    if gaussians.rgbdecoder is not None:
        gaussians.rgbdecoder.cuda()
        gaussians.rgbdecoder.eval()

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, scaling_modifier=1.0,
                           basicfunction=rbfbasefunction,  GRsetting=GRsetting, GRzer=GRzer)["render"]  # C x H x W
        render_paths = save_path_template.makedirs(render_path, view)
        torchvision.utils.save_image(
            rendering, render_paths.full)


def run_test(dataset: ModelParams, pipeline: PipelineParams, test_iteration: int, skip_train: bool, skip_test: bool, multiview: bool, duration: int,
             rgbfunction="rgbv1", rdpip="v2", loader="colmap", cameras_validate_all=False, *_args, **_kwargs):

    with torch.no_grad():
        print("use model {}".format(dataset.model))
        # default, gmodel, we are tewsting
        GaussianModel = getmodel(dataset.model)

        gaussians = GaussianModel(dataset.sh_degree, rgbfunction)

        scene = Scene(dataset, gaussians, load_iteration=test_iteration,
                      shuffle=False, multiview=multiview, duration=duration, loader=loader)
        rbfbasefunction = trbfunction
        numchannels = 9
        bg_color = [0 for _ in range(numchannels)]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if gaussians.ts is None:
            cameraslit = scene.getTestCameras()
            H, W = cameraslit[0].image_height, cameraslit[0].image_width
            gaussians.ts = torch.ones(1, 1, H, W).cuda()

        if skip_test:
            print("Skipping test rendering as per user request.")
            return

        if cameras_validate_all:
            views = list(scene.getAllCameras())
            save_path_template = SavePathTemplate(
                duration=duration, by_image_name_and_timestamp=True)
            name_suffix = "_validate_all"
        else:
            views = list(scene.getTestCameras())
            save_path_template = SavePathTemplate(
                duration=duration, by_image_name_and_timestamp=False)
            name_suffix = ""

        if multiview:
            render_setnogt(
                dataset.model_path, f"mv{name_suffix}", scene.loaded_iter,
                views, gaussians, pipeline, background, rbfbasefunction, rdpip, save_path_template)
        else:
            render_set(
                dataset.model_path, f"test{name_suffix}", scene.loaded_iter,
                views, gaussians, pipeline, background, rbfbasefunction, rdpip, save_path_template)


if __name__ == "__main__":
    args, model_extract, pp_extract, _ = gettestparse()
    run_test(dataset=model_extract, pipeline=pp_extract, **vars(args))
