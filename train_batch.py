#!/bin/python
from dataclasses import dataclass
import itertools
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Iterable, Literal, Optional, final
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
        return cls(parts[0], [p for p in parts[1:] if p.strip()])

    def items(self):
        for i in self.values:
            yield self.name, i


def eval_iter(arg: str):
    match arg:
        case "zip":
            return zip
        case "product":
            return itertools.product
        case _:
            raise tap.ArgumentTypeError(
                f"Invalid value for --iter-parameters: {arg}. Must be 'zip' or 'product'.")


class TrainBatch(tap.Tap):
    DataNames: Optional[list[str]] = None
    DataName: Optional[str] = None
    ExperimentName: str = ""
    parameters: list[Parameter]
    iter_parameters: Callable[[Iterable], Iterable]
    config_base: Path = Path("configs/techni_lite/noprior48.json")
    dryrun: bool = False

    def configure(self):
        self.add_argument("-p", "--parameters",
                          type=Parameter.from_string, action="append")
        self.add_argument("-i", "--iter-parameters", type=eval_iter,
                          choices=[eval_iter(i) for i in ["zip", "product"]], default=eval_iter("product"))
        self.add_argument("-n", "--dryrun", action="store_true", default=False)

    def process_args(self):
        if self.DataNames is None and self.DataName is not None:
            self.DataNames = [self.DataName]
        if self.DataNames is None:
            raise tap.ArgumentError(
                self.DataNames, "Provide either DataNames or DataName.")

    @final
    def train_batch(self):
        def throw_on_error(cmd: str):
            error = os.system(cmd)
            if error != 0:
                raise ChildProcessError(
                    f"Command '{cmd}' failed with error code {error}")

        run_batch: Callable[[str], None] = \
            print if self.dryrun else throw_on_error
        pairs: list[tuple[str, str]]
        for DataName in self.DataNames or []:
            for pairs in self.iter_parameters(*[p.items() for p in self.parameters]):
                names = "_".join([p[0] for p in pairs])
                values = "_".join([p[1] for p in pairs])
                ExperimentName = f'_{self.ExperimentName}' if self.ExperimentName else ''
                SrcPath = f"/home/aq85800/NewVolume/SpacetimeGaussians/blender_prior/{DataName}/point/colmap_0"
                ModelPath = f"/home/aq85800/NewVolume/SpacetimeGaussians/output_sweep_tests/{DataName}{ExperimentName}/{names}/{values}"
                params = " ".join(
                    [f"--{name} {value}" for name, value in pairs])
                command = f"""
    python -m cProfile -o ~/NewVolume/SpacetimeGaussians/profile/{DataName}.cprofile train.py --eval --config {self.config_base} -s {SrcPath} -m {ModelPath} {params} \\
    && python test.py --eval --configpath configs/techni_lite/noprior48.json -oc test_iteration=30000 --valloader technicolor --skip_train -s {SrcPath} -m {ModelPath}
    """.strip()
                run_batch(command)


if __name__ == "__main__":
    TrainBatch().parse_args().train_batch()
