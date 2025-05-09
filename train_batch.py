#!/bin/python
import os
import tap


def train_batch(DataName: str, dryrun: bool = False):
    for reg in (0, 1, 9, 10, 4, 5, 6, 7, 8):
        SrcPath = f"/home/aq85800/NewVolume/SpacetimeGaussians/blender_prior/{DataName}/point/colmap_0"
        ModelPath = f"/home/aq85800/NewVolume/SpacetimeGaussians/output_sweep_tests/{DataName}/reg/{reg}"
        command = f"""
python -m cProfile -o ~/NewVolume/SpacetimeGaussians/profile/{DataName}.cprofile train.py --eval --config configs/techni_lite/noprior48.json -s {SrcPath} -m {ModelPath} --reg {reg} \\
&& python test.py --eval --configpath configs/techni_lite/noprior48.json -oc test_iteration=30000 --valloader technicolor --skip_train -s {SrcPath} -m {ModelPath}
""".strip()
        if dryrun:
            print(command)
        else:
            os.system(command)


if __name__ == "__main__":
    tap.tapify(train_batch)
