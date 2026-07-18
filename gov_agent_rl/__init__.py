from pathlib import Path

_src_package = Path(__file__).resolve().parent.parent / "src" / "gov_agent_rl"
if _src_package.exists():
    __path__.append(str(_src_package))

__version__ = "0.1.0"
