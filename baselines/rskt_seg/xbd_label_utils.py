"""Rasterize xBD/xView2 building polygons as binary masks.

The implementation follows the preprocessing used by SegEarth-OV:
``features.xy[*].wkt`` polygon exteriors are rounded to integer coordinates
and filled with OpenCV. Every annotated polygon is assigned building ID 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _fallback_polygon_exterior(wkt_string: str) -> np.ndarray:
    """Parse the standard xBD ``POLYGON ((x y, ...))`` exterior ring."""
    text = wkt_string.strip()
    if not text.upper().startswith("POLYGON"):
        raise ValueError(f"unsupported xBD WKT geometry: {text[:32]!r}")
    try:
        inner = text[text.index("((") + 2 : text.rindex("))")]
        # Match the official converter, which uses only poly.exterior.coords.
        inner = inner.split("),", maxsplit=1)[0]
        coordinate_pairs = []
        for point in inner.split(","):
            values = point.strip().split()
            if len(values) < 2:
                raise ValueError(f"invalid coordinate: {point!r}")
            coordinate_pairs.append([float(values[0]), float(values[1])])
        coordinates = np.asarray(coordinate_pairs, dtype=np.float32)
    except (TypeError, ValueError, IndexError) as error:
        raise ValueError(f"cannot parse xBD polygon WKT: {text[:80]!r}") from error
    if (
        coordinates.ndim != 2
        or coordinates.shape[0] < 3
        or coordinates.shape[1] != 2
    ):
        raise ValueError(
            f"xBD polygon has invalid exterior shape: {coordinates.shape}"
        )
    return coordinates


def polygon_exterior_from_wkt(wkt_string: str) -> np.ndarray:
    """Return the float32 ``[N,2]`` exterior ring of an xBD polygon."""
    try:
        from shapely import wkt as shapely_wkt  # type: ignore
    except ImportError:
        return _fallback_polygon_exterior(wkt_string)

    try:
        geometry = shapely_wkt.loads(wkt_string)
        coordinates = np.asarray(geometry.exterior.coords, dtype=np.float32)
    except Exception as error:
        raise ValueError(
            f"cannot parse xBD polygon WKT: {wkt_string[:80]!r}"
        ) from error
    if (
        coordinates.ndim != 2
        or coordinates.shape[0] < 3
        or coordinates.shape[1] != 2
    ):
        raise ValueError(
            f"xBD polygon has invalid exterior shape: {coordinates.shape}"
        )
    return coordinates


def load_xbd_building_mask(
    json_path: str | Path,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterize one xBD annotation as a binary ``uint8`` building mask."""
    import cv2

    path = Path(json_path)
    with path.open("r", encoding="utf-8") as file:
        annotation = json.load(file)

    features = annotation.get("features", {}).get("xy", [])
    if not isinstance(features, list):
        raise ValueError(f"xBD features.xy must be a list: {path}")

    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    for feature_index, feature in enumerate(features):
        if not isinstance(feature, dict):
            raise ValueError(
                f"xBD feature {feature_index} is not an object: {path}"
            )
        wkt_string = feature.get("wkt", "")
        if not isinstance(wkt_string, str) or not wkt_string.strip():
            raise ValueError(
                f"xBD feature {feature_index} has no valid WKT: {path}"
            )
        try:
            exterior = polygon_exterior_from_wkt(wkt_string)
        except ValueError as error:
            raise ValueError(
                f"failed to parse xBD feature {feature_index}: {path}"
            ) from error
        points = np.round(exterior).astype(np.int32)
        cv2.fillPoly(mask, [points], color=1)

    return mask
