#!/bin/python
from dataclasses import dataclass
import glob
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
    SourceDir: Path
    '''base directory for source data, aka parent directory of DataNames'''
    ModelDir: Path
    '''base directory for model output, aka parent directory of {DataNames}_{ExperimentName}'''
    DataNames: Optional[list[str]] = None
    '''list of directory names or glob patterns relative to --SourceDir. If glob patterns are used, they must be quoted, as to prevent shell expansion in the wrong directory.'''
    DataName: Optional[str] = None
    '''Single directory version of --DataNames, for backward compatibility. Ignored if --DataNames is provided.'''
    ExperimentName: str = ""
    parameters: list[Parameter]
    iter_parameters: Callable[[Iterable], Iterable]
    config_base: Path = Path("configs/techni_lite/noprior48.json")
    dryrun: bool = False
    _expanded_data: list[Path] = []
    """Expanded DataNames, populated after processing args"""

    def configure(self):
        self.add_argument("-p", "--parameters",
                          type=Parameter.from_string, action="append", default=[])
        self.add_argument("-i", "--iter-parameters", type=eval_iter,
                          choices=[eval_iter(i) for i in ["zip", "product"]], default=eval_iter("product"))
        self.add_argument("-n", "--dryrun", action="store_true", default=False)

    def process_args(self):
        if self.DataNames is None and self.DataName is not None:
            self.DataNames = [self.DataName]
        if self.DataNames is None:
            raise SystemExit(
                '--(DataNames|DataName): One of these must be provided.')
        self.SourceDir = self.SourceDir.expanduser()
        self.ModelDir = self.ModelDir.expanduser()
        self._expanded_data = list(itertools.chain.from_iterable(
            (self.SourceDir.glob(n) for n in self.DataNames)))
        if len(self._expanded_data) == 0:
            raise SystemExit(
                "--(DataNames|DataName): No matching data directories found relative to --SourceDir.")

    def train_batch(self):
        def throw_on_error(cmd: str):
            error = os.system(cmd)
            if error != 0:
                raise ChildProcessError(
                    f"Command '{cmd}' failed with error code {error}")
        run_batch: Callable[[str], None] = \
            print if self.dryrun else throw_on_error
        for DataName in self._expanded_data:
            pairs: list[tuple[str, str]]
            for pairs in self.iter_parameters(*[p.items() for p in self.parameters]):
                names = "+".join([p[0] for p in pairs])
                values = "+".join([p[1] for p in pairs])
                SrcPath = DataName/'point'/'colmap_0'
                ModelPath = self.ModelDir/DataName.stem/self.ExperimentName/names/values
                params = " ".join(
                    [f"--{name} {value}" for name, value in pairs])
                command = f"""
    python train.py --eval --config {self.config_base} -s {SrcPath} -m {ModelPath} {params} \\
    && python test.py --eval --configpath configs/techni_lite/noprior48.json -oc test_iteration=30000 --valloader technicolor --skip_train -s {SrcPath} -m {ModelPath}
    """.strip()
                run_batch(command)


if __name__ == "__main__":
    TrainBatch().parse_args().train_batch()
