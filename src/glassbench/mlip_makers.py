from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any

import yaml
from atomate2.forcefields.jobs import ForceFieldRelaxMaker, ForceFieldStaticMaker
from atomate2.forcefields.md import ForceFieldMDMaker

logger = logging.getLogger(__name__)


def get_mlip_params_path():
    """Helper to resolve the parameters directory."""
    env_path = os.environ.get("MLIP_PARAMS_DIR")
    if not env_path:
        raise OSError(
            "MLIP_PARAMS_DIR environment variable is not set. "
            "Point it at the directory containing potentials.yaml."
        )
    return env_path


def available_potentials(potential_name: str = None,
                         functional: str = "pbe",
                         size: str = "large",
                         model_type: str = "grace-fs",
                         architecture: str = "cpu") -> dict | None:

    params_dir = get_mlip_params_path()
    yaml_path = os.path.join(params_dir, "potentials.yaml")

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Could not find potentials.yaml at {yaml_path}. "
                                "Please set MLIP_PARAMS_DIR environment variable.")

    with open(yaml_path) as f:
        available_pots = yaml.safe_load(f)

    # Check by name first
    for name, information in available_pots.items():
        if name == potential_name:
            if architecture in information:
                logger.debug("Found potential '%s' by name (%s, %s)",
                             name, information.get("model_type"), information.get("functional"))
                return information
            raise ValueError(f"Architecture '{architecture}' not available for '{name}'")

    # Check by properties
    for name, information in available_pots.items():
        if (information.get("functional") == functional and
                information.get("size") == size and
                information.get("model_type") == model_type):
            if architecture in information:
                logger.debug("Found potential '%s' by properties (%s, %s, %s)",
                             name, model_type, functional, size)
                return information
            raise ValueError(f"Architecture '{architecture}' not available for '{name}'")

    logger.warning("No potential found matching name=%r functional=%s size=%s model_type=%s",
                   potential_name, functional, size, model_type)
    return None



_CALCULATOR_BUILDERS: dict[str, Callable[..., object]] = {}

def register_calculator(*model_types: str):
    """Decorator: register an ASE-calculator builder for one or more model types.

    Example
    -------
    @register_calculator("my-mlip")
    def _build_my_mlip(model_path, **kwargs):
        from my_mlip.ase import MyCalculator
        return MyCalculator(model_path, **kwargs)
    """
    def decorator(builder: Callable[..., object]) -> Callable[..., object]:
        for model_type in model_types:
            _CALCULATOR_BUILDERS[model_type] = builder
        return builder
    return decorator


@register_calculator("grace-fs")
def _build_grace_fs(model_path, **kwargs):
    from pyace.asecalc import PyGRACEFSCalculator
    return PyGRACEFSCalculator(model_path, **kwargs)

@register_calculator("grace")
def _build_grace(model_path, **kwargs):
    from tensorpotential.calculator import TPCalculator
    return TPCalculator(model_path, **kwargs)

@register_calculator("mace")
def _build_mace(model_path, **kwargs):
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=model_path, **kwargs)

@cache
def _cached_calculator(model_path, model_type, builder_kwargs_items):
    logger.info("Loading %s calculator from %s", model_type, model_path)
    builder = _CALCULATOR_BUILDERS[model_type]
    calc = builder(model_path, **dict(builder_kwargs_items))
    logger.info("Calculator ready (%s)", model_type)
    return calc

def make_calculator(builder_kwargs: dict | None = None, **selection_kwargs: Any) -> Any:
    builder_kwargs = builder_kwargs or {}
    architecture = selection_kwargs.get("architecture", "cpu")

    potential = available_potentials(**selection_kwargs)
    if potential is None:
        raise ValueError(f"Could not find a valid potential for: {selection_kwargs}")

    model_type = potential.get("model_type")
    if model_type not in _CALCULATOR_BUILDERS:
        raise ValueError(f"No builder registered for '{model_type}'. "
                         f"Registered: {sorted(_CALCULATOR_BUILDERS)}")
    if architecture not in potential:
        raise ValueError(f"Architecture '{architecture}' not available for '{model_type}'.")

    model_path = os.path.join(get_mlip_params_path(), potential[architecture])
    # builder_kwargs must be hashable for the cache (e.g. {"device": "cuda"})
    return _cached_calculator(model_path, model_type, tuple(sorted(builder_kwargs.items())))


# ---------------------------------------------------------------------------
# Makers
# ---------------------------------------------------------------------------

class _MLIPCalculatorMixin:
    """Shared ``calculator`` property for all MLIP makers.
    Defined as a plain (non-dataclass) mixin so it adds no fields and avoids
    dataclass field-ordering issues with the atomate2 base classes.

    Calculator priority:
    1. ``calculator_instance`` — an ASE calculator passed in directly.
    2. ``potential_kwargs`` — name/property lookup via potentials.yaml.
    """

    @property
    def calculator(self):
        if getattr(self, "calculator_instance", None) is not None:
            return self.calculator_instance
        return make_calculator(**(self.potential_kwargs or {}))


@dataclass
class MLIPRelaxMaker(_MLIPCalculatorMixin, ForceFieldRelaxMaker):
    name: str = "RelaxMaker"
    potential_kwargs: dict = None
    calculator_instance: Any = None


@dataclass
class MLIPStaticMaker(_MLIPCalculatorMixin, ForceFieldStaticMaker):
    name: str = "StaticMaker"
    potential_kwargs: dict = None
    calculator_instance: Any = None


@dataclass
class MLIPMDMaker(_MLIPCalculatorMixin, ForceFieldMDMaker):
    name: str = "MLIPMDMaker"
    potential_kwargs: dict = None
    calculator_instance: Any = None