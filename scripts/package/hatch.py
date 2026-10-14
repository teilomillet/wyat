"""Build the runtime wheel from the reviewed public source inventory."""

import sys
from pathlib import Path, PurePosixPath

from hatchling.builders.config import BuilderConfig
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# Hatchling loads this file under its configured path, whose dotted filename
# gives it a nonempty __package__ even outside an importable source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.package.check import read_configuration
from scripts.package.privacy import check_files, check_text


class CustomBuildHook(BuildHookInterface[BuilderConfig]):
    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        root = Path(self.root)
        configuration = read_configuration(root)
        selected = configuration["tool"]["hatch"]["build"]["targets"]["sdist"][
            "only-include"
        ]
        if not isinstance(selected, list) or not selected:
            raise ValueError("A nonempty public source inventory is required")
        runtime: dict[str, str] = {}
        seen: set[str] = set()
        for name in selected:
            if not isinstance(name, str):
                raise ValueError("Public source entries must be literal file names")
            check_text("pyproject.toml", name.encode("utf-8"))
            parts = PurePosixPath(name).parts
            if (
                not parts
                or PurePosixPath(name).is_absolute()
                or any(part in {"", ".", ".."} for part in name.split("/"))
                or any(character in name for character in "\\:*?[]\x00")
                or name in seen
            ):
                raise ValueError(f"Invalid public source entry: {name!r}")
            seen.add(name)
            source = root / name
            if not source.is_file() or any(
                (root.joinpath(*parts[:index])).is_symlink()
                for index in range(1, len(parts) + 1)
            ):
                raise ValueError(f"Missing or symbolic public source file: {name}")
            if name.startswith("wyat/"):
                runtime[str(source)] = name
        if not runtime:
            raise ValueError("Public source inventory omits the runtime")
        check_files({name: root / name for name in seen})
        if self.target_name == "wheel" and version != "editable":
            build_data["force_include"] = runtime
