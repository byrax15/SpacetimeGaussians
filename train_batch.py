#!/bin/python
from dataclasses import dataclass
import itertools as it
import os
from typing import Optional, final
import tap


@dataclass
class Parameter:
    name: str
    values: list[str]

    @classmethod
    def from_string(cls, args: str):
        parts = args.split(":")
        if len(parts) < 2:
            raise ValueError(
                "Parameter string must be in the format 'name(:valueN)+'")
        return cls(parts[0], parts[1:])

    def items(self):
        for i in self.values:
            yield self.name, i


class TrainBatch(tap.Tap):
    DataNames: Optional[list[str]] = None
    DataName: Optional[str] = None
    parameters: list[Parameter]
    dryrun: bool = False,

    def configure(self):
        self.add_argument("-p", "--parameters",
                          type=Parameter.from_string, action="append")
        self.add_argument("-n", "--dryrun", action="store_true", default=False)
        pass

    def process_args(self):
        if self.DataNames is None and self.DataName is not None:
            self.DataNames = [self.DataName]
        if self.DataNames is None:
            raise tap.ArgumentError("Provide either DataNames or DataName.")

    @final
    def train_batch(self):
        run_batch: callable[[str], None] = print if self.dryrun else os.system
        pairs: list[tuple[str, str]]
        for DataName, *pairs in it.product(self.DataNames, *[p.items() for p in self.parameters]):
            names = "_".join([p[0] for p in pairs])
            values = "_".join([p[1] for p in pairs])
            SrcPath = f"/home/aq85800/NewVolume/SpacetimeGaussians/blender_prior/{DataName}/point/colmap_0"
            ModelPath = f"/home/aq85800/NewVolume/SpacetimeGaussians/output_sweep_tests/{DataName}/{names}/{values}"
            params = " ".join([f"--{name} {value}" for name, value in pairs])
            command = f"""
python -m cProfile -o ~/NewVolume/SpacetimeGaussians/profile/{DataName}.cprofile train.py --eval --config configs/techni_lite/noprior48.json -s {SrcPath} -m {ModelPath} {params} \\
&& python test.py --eval --configpath configs/techni_lite/noprior48.json -oc test_iteration=30000 --valloader technicolor --skip_train -s {SrcPath} -m {ModelPath}
""".strip()
            run_batch(command)


if __name__ == "__main__":
    TrainBatch().parse_args().train_batch()
