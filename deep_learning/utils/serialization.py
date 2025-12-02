# -*- coding: utf-8 -*-
"""
模型保存/加载工具

支持:
- save_model/load_model: pickle 或 JSON 格式

English:
- save_model/load_model with pickle or JSON.
"""

from __future__ import annotations

import importlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


def _to_serializable(obj: Any):
    """递归将对象转换为 JSON 可序列化的结构。"""
    if np is not None and hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def save_model(model: Any, path: str | Path, fmt: str = "pickle") -> None:
    """
    保存模型到文件。

    Args:
        model: 任意 Python 对象
        path: 保存路径
        fmt: 'pickle' 或 'json'
    """
    path = Path(path)
    fmt = fmt.lower()

    if fmt == "pickle":
        with path.open("wb") as f:
            pickle.dump(model, f)
        return

    if fmt != "json":
        raise ValueError("fmt 仅支持 'pickle' 或 'json'")

    state = _to_serializable(getattr(model, "__dict__", {}))
    payload: Dict[str, Any] = {
        "class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "state": state,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_model(path: str | Path, fmt: str = "pickle") -> Any:
    """
    从文件加载模型。

    Args:
        path: 文件路径
        fmt: 'pickle' 或 'json'
    """
    path = Path(path)
    fmt = fmt.lower()

    if fmt == "pickle":
        with path.open("rb") as f:
            return pickle.load(f)

    if fmt != "json":
        raise ValueError("fmt 仅支持 'pickle' 或 'json'")

    payload = json.loads(path.read_text(encoding="utf-8"))
    cls_path = payload["class"]
    module_name, class_name = cls_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    instance = cls.__new__(cls)
    instance.__dict__.update(payload["state"])
    return instance


__all__ = ["save_model", "load_model"]
