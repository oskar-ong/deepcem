from ..config import AlgoConfig
from .base import Strategy
from .late_fusion import LateFusion
from .inline_encode import InlineEncode

# map "strategy id" -> class
_STRATEGY_REGISTRY: dict[int, type[Strategy]] = {
    1: LateFusion,
    2: InlineEncode,
}

# map "strategy id" -> *instance*
_STRATEGY_INSTANCES: dict[int, Strategy] = {}

def strategy_factory(configuration: AlgoConfig) -> Strategy:
    """
    Return a Strategy instance:
      - first call for a given configuration.strategy => create and cache
      - later calls with same configuration.strategy => return cached
    """
    key = configuration.strategy  
    if key in _STRATEGY_INSTANCES:
        return _STRATEGY_INSTANCES[key]

    try:
        cls = _STRATEGY_REGISTRY[key]
    except KeyError:
        raise ValueError(f"Unknown strategy: {key}")

    instance = cls()
    _STRATEGY_INSTANCES[key] = instance
    return instance
