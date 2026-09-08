"""Load the packaged, human-readable policy used by the demo."""

import tomllib
from importlib.resources import files


def load_policy() -> dict:
    return tomllib.loads(files("manto").joinpath("demo_policy.toml").read_text(encoding="utf-8"))
