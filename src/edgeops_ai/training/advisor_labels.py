"""Apply reviewed annotations to an existing Advisor export without querying InfluxDB."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from edgeops_ai.training.advisor_dataset import (
    EVENT_COLUMNS,
    KINDS,
    SCHEMA,
    load_manual_labels,
    timestamp,
)
from edgeops_ai.training.reports import read_report, write_report


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def apply_reviews(input_dir: Path, labels_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("Output already exists; choose a new dataset directory")
    card_path = input_dir / "data-card.json"
    card = read_report(card_path)
    if card.get("schema_version") != "advisor-1":
        raise ValueError("Expected an advisor-1 dataset")
    start, stop = timestamp(card["start"]), timestamp(card["stop"])
    train_stop = timestamp(card["split"]["train_stop"])
    validation_stop = timestamp(card["split"]["validation_stop"])
    if not start < train_stop < validation_stop < stop:
        raise ValueError("Invalid source dataset time boundaries")
    previous = load_manual_labels(input_dir / "manual-labels.csv", start, stop)
    incoming = load_manual_labels(labels_path, start, stop)
    indexed = {
        (r["entity_type"], r["entity_id"], r["start"], r["stop"]): r for r in [*previous, *incoming]
    }
    labels = [indexed[key] for key in sorted(indexed)]
    if not labels:
        raise ValueError("No reviewed annotations supplied")
    for index, label in enumerate(labels):
        if any(label["start"] < cut < label["stop"] for cut in (train_stop, validation_stop)):
            raise ValueError("Split annotation intervals at dataset split boundaries")
        for other in labels[:index]:
            same_entity = label["entity_id"] == other["entity_id"] or "*" in {
                label["entity_id"],
                other["entity_id"],
            }
            if (
                same_entity
                and label["entity_type"] == other["entity_type"]
                and max(label["start"], other["start"]) < min(label["stop"], other["stop"])
                and (label["label"] != other["label"] or label["entity_id"] == other["entity_id"])
            ):
                raise ValueError("Conflicting or overlapping reviewed annotations")
    with (input_dir / "weak-labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(EVENT_COLUMNS).issubset(reader.fieldnames or []):
            raise ValueError("Invalid weak-label event CSV")
        gaps = [
            {**row, "start": timestamp(row["start"]), "stop": timestamp(row["stop"])}
            for row in reader
            if row["label"] in {"sensor-telemetry-gap", "all-devices-silent"}
        ]
    matched: set[int] = set()
    row_counts: Counter[str] = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".advisor-reviews-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        for kind in KINDS:
            path = input_dir / f"{kind}.parquet"
            if file_hash(path) != card["tables"][kind]["sha256"]:
                raise ValueError(f"Source Parquet hash mismatch: {kind}")
            parquet = pq.ParquetFile(path)
            if (
                parquet.schema_arrow != SCHEMA
                or parquet.metadata.num_rows != card["tables"][kind]["rows"]
            ):
                raise ValueError(f"Source Parquet schema or row count mismatch: {kind}")
            relevant = [
                (i, label) for i, label in enumerate(labels) if label["entity_type"] == kind
            ]
            labelled_rows = 0
            with pq.ParquetWriter(staging / path.name, SCHEMA) as writer:
                for batch in parquet.iter_batches(batch_size=65536):
                    table = pa.Table.from_batches([batch])
                    manual = table["manual_label"]
                    for index, label in relevant:
                        begin = pa.scalar(label["start"], type=SCHEMA.field("time").type)
                        end = pa.scalar(label["stop"], type=SCHEMA.field("time").type)
                        mask = pc.and_(
                            pc.greater_equal(table["time"], begin), pc.less(table["time"], end)
                        )
                        if label["entity_id"] != "*":
                            mask = pc.and_(mask, pc.equal(table["entity_id"], label["entity_id"]))
                        if pc.any(mask).as_py():
                            matched.add(index)
                            manual = pc.if_else(mask, label["label"], manual)
                    for count in pc.value_counts(manual).to_pylist():
                        if count["values"] is not None:
                            row_counts[count["values"]] += count["counts"]
                            labelled_rows += count["counts"]
                    table = table.set_column(
                        table.schema.get_field_index("manual_label"),
                        SCHEMA.field("manual_label"),
                        manual,
                    )
                    writer.write_table(table)
            card["tables"][kind]["sha256"] = file_hash(staging / path.name)
            card["tables"][kind]["manual_label_rows"] = labelled_rows
            print(f"Reviewed {kind}: {labelled_rows} labelled rows", flush=True)
        gap_only = 0
        for index, label in enumerate(labels):
            if index not in matched:
                if not any(
                    gap["entity_type"] == label["entity_type"]
                    and gap["entity_id"] == label["entity_id"]
                    and gap["start"] <= label["start"] < label["stop"] <= gap["stop"]
                    for gap in gaps
                ):
                    raise ValueError(
                        f"Review matches neither measurements nor a gap: {label['entity_id']}"
                    )
                gap_only += 1
        shutil.copyfile(card_path, staging / "source-data-card.json")
        shutil.copyfile(input_dir / "data-card.md", staging / "source-data-card.md")
        shutil.copyfile(input_dir / "weak-labels.csv", staging / "weak-labels.csv")
        with (staging / "manual-labels.csv").open("w", encoding="utf-8", newline="") as handle:
            writer_csv = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS, extrasaction="ignore")
            writer_csv.writeheader()
            writer_csv.writerows(labels)
        by_split = Counter(
            "train"
            if label["start"] < train_stop
            else "validation"
            if label["start"] < validation_stop
            else "test"
            for label in labels
        )
        card["manual_labels"] = {
            "reviewed_intervals": len(labels),
            "input_sha256": file_hash(labels_path),
        }
        card["review"] = {
            "source_card_sha256": file_hash(card_path),
            "by_split": dict(by_split),
            "by_label": dict(Counter(r["label"] for r in labels)),
            "labelled_rows": dict(row_counts),
            "gap_only_intervals": gap_only,
        }
        write_report(card, staging / "data-card.json")
        (staging / "data-card.md").write_text(
            "# Reviewed Advisor dataset\n\n"
            f"Applied {len(labels)} reviewed intervals: {dict(by_split)}. "
            f"{gap_only} intervals describe missing telemetry and have no measurement rows.\n\n"
            "The original measurements, row order, cadence, splits and weak labels "
            "remain unchanged. "
            "For reviewed measurement windows, manual_label takes precedence over weak_labels. "
            "Gap-only reviews remain in manual-labels.csv; no data is filled or fabricated. "
            "Wildcard entity_id=* covers all entities of the specified type.\n\n"
            "Reviews include explicit corrections to training-period weak events and a separate "
            "small validation set. They are human reports, "
            "not independently proven incident causes. "
            "Review notes retain timing uncertainty. Unreviewed windows remain unlabelled.\n\n"
            "manual-labels.csv records current reviews, including retained previous intervals. "
            "weak-labels.csv retains the source weak events. data-card.json records updated "
            "hashes and review counts; source-data-card.json/md preserve source provenance "
            "and coverage/quality details. "
            "Earlier verification reports describe the source only.\n",
            encoding="utf-8",
        )
        staging.rename(output)
    return card


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        card = apply_reviews(args.input_dir, args.labels, args.output_dir)
        print(f"Dataset: {args.output_dir} ({card['manual_labels']['reviewed_intervals']} reviews)")
        return 0
    except (OSError, ValueError, KeyError, TypeError, pa.ArrowException) as error:
        print(f"Advisor review import failed: {error}")
        return 1
