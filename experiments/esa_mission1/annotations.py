"""Read ESA annotation tables for evaluation after detector scores are frozen."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ingest import CHANNELS, utc_index

QUALITY = {"Communication Gap", "Invalid Data"}


@dataclass(frozen=True)
class Annotation:
    identity: str
    category: str
    start: pd.Timestamp
    end: pd.Timestamp


def load_annotations(labels_stream, types_stream):
    labels = pd.read_csv(labels_stream, dtype={"ID": str})
    types = pd.read_csv(types_stream, dtype={"ID": str})
    if not {"ID", "Channel", "StartTime", "EndTime"} <= set(labels) or not {"ID", "Category"} <= set(types):
        raise ValueError("Unexpected ESA label/type columns")
    if types["ID"].isna().any() or types["ID"].duplicated().any() or types["Category"].isna().any():
        raise ValueError("Missing or duplicate ESA annotation type keys")
    rows = labels.loc[labels["Channel"].isin(CHANNELS)].merge(
        types[["ID", "Category"]], on="ID", how="left", validate="many_to_one"
    )
    if rows["Category"].isna().any():
        raise ValueError("An ESA annotation has no linked category")
    allowed = QUALITY | {"Anomaly", "Rare Event"}
    if not set(rows["Category"]) <= allowed:
        raise ValueError(f"Unexpected ESA annotation categories: {set(rows['Category']) - allowed}")
    rows = rows.assign(StartTime=utc_index(rows["StartTime"]), EndTime=utc_index(rows["EndTime"]))
    if (rows["EndTime"] < rows["StartTime"]).any():
        raise ValueError("Reversed ESA annotation time bounds")
    return {
        channel: [Annotation(row.ID, row.Category, row.StartTime, row.EndTime)
                  for row in rows.loc[rows["Channel"] == channel].itertuples(index=False)]
        for channel in CHANNELS
    }


def evaluation_inputs(grid, eligible, emitted, annotations):
    """Censor quality periods retrospectively without changing detector state."""
    observable = eligible.copy()
    rare = np.zeros(len(grid), dtype=bool)
    events = {}
    for annotation in annotations:
        label_mask = (grid >= annotation.start) & (grid <= annotation.end)
        if annotation.category in QUALITY:
            observable &= ~label_mask
        elif annotation.category == "Rare Event":
            rare |= label_mask
        else:
            events.setdefault(annotation.identity, np.zeros(len(grid), dtype=bool))
            events[annotation.identity] |= label_mask
    return observable, emitted & observable, rare & observable, {
        key: indices & observable for key, indices in events.items()
    }
