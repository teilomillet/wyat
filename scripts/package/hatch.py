"""Build the runtime wheel from the reviewed public source inventory."""

import tomllib
from pathlib import Path, PurePosixPath

from hatchling.builders.config import BuilderConfig
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface[BuilderConfig]):
    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        root = Path(self.root)
        configuration = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))
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
        if self.target_name == "wheel":
            build_data["force_include"] = runtime
