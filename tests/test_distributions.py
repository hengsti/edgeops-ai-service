import numpy as np
import pytest

from edgeops_ai.ml import MODEL_FEATURE_COLUMNS, vectorize_model_features


def matrix(make_features, **overrides):
    features = make_features(**overrides)
    return np.asarray([[*vectorize_model_features(features), features.processing_ratio]] * 30)


def test_identical_distributions_pass(make_features):
    from edgeops_ai.training.distributions import build_distribution_report

    values = matrix(make_features)
    report = build_distribution_report(values, values)
    assert report["passed"]
    assert len(report["features"]) == len(MODEL_FEATURE_COLUMNS) + 1
    assert all(item["ks_distance"] == 0 for item in report["features"].values())


@pytest.mark.parametrize(
    "overrides, failed",
    [
        ({"processing_ratio": 1.2}, "raw_processing_ratio"),
        ({"queue_utilization_ratio": None}, "queue_metrics_available"),
    ],
)
def test_large_skew_including_unclipped_ratios_and_missingness_fails(
    make_features, overrides, failed
):
    from edgeops_ai.training.distributions import build_distribution_report

    report = build_distribution_report(matrix(make_features), matrix(make_features, **overrides))
    assert not report["passed"]
    assert not report["features"][failed]["passed"]
    assert report["features"][failed]["ks_distance"] == 1


@pytest.mark.parametrize("rows", [0, 1, 19])
def test_distribution_check_rejects_too_few_windows(make_features, rows):
    from edgeops_ai.training.distributions import build_distribution_report

    with pytest.raises(ValueError, match="20"):
        build_distribution_report(matrix(make_features), matrix(make_features)[:rows])


def test_release_gate_recomputes_results_instead_of_trusting_passed_flag(make_features):
    from edgeops_ai.training.distributions import build_distribution_report, distribution_passed

    report = build_distribution_report(matrix(make_features), matrix(make_features))
    assert distribution_passed(report)
    report["features"]["raw_processing_ratio"]["ks_distance"] = 1.0
    assert not distribution_passed(report)
    assert not distribution_passed({"passed": True})


def test_production_replay_uses_current_extractor(production_snapshots):
    from edgeops_ai.training.distributions import load_production_features

    features, source = load_production_features(production_snapshots)
    assert len(features) == 30
    assert source["observations"] == 31
    assert all(row.processing_ratio == 1 and row.queue_growth_rate == 0 for row in features)


def test_production_replay_rejects_simulation_provenance(production_snapshots):
    from edgeops_ai.training.distributions import load_production_features

    production_snapshots.write_text(
        production_snapshots.read_text().replace('"simulated":false', '"simulated":true')
    )
    with pytest.raises(ValueError, match="real production"):
        load_production_features(production_snapshots)


def test_outside_range_limit_can_fail_even_with_a_small_ks_distance(make_features):
    from edgeops_ai.training.distributions import build_distribution_report

    reference = matrix(make_features)
    observed = reference.copy()
    observed[:4, -1] = 1.2
    report = build_distribution_report(reference, observed)
    feature = report["features"]["raw_processing_ratio"]
    assert feature["ks_distance"] < report["thresholds"]["max_ks_distance"]
    assert not feature["passed"]


def test_release_gate_rejects_missing_or_changed_evidence(tmp_path, make_features):
    import hashlib

    from edgeops_ai.training.distributions import build_distribution_report, simulation_digest
    from edgeops_ai.training.reports import write_report
    from edgeops_ai.training.validate import append_distribution_gate

    simulation = tmp_path / "normal-seed-42.csv"
    simulation.write_text("simulation inputs", encoding="utf-8")
    production = tmp_path / "production.jsonl"
    production.write_text("production inputs", encoding="utf-8")
    path = tmp_path / "distribution-report.json"
    summary = {"gates": [], "passed": True}
    append_distribution_gate(summary, path, training_dir=tmp_path)
    assert not summary["passed"]
    report = build_distribution_report(matrix(make_features), matrix(make_features))
    report["simulation"] = {"path": str(tmp_path), "sha256": simulation_digest(tmp_path)}
    report["production"] = {
        "path": str(production),
        "sha256": hashlib.sha256(production.read_bytes()).hexdigest(),
    }
    write_report(report, path)
    summary = {"gates": [], "passed": True}
    append_distribution_gate(summary, path, training_dir=tmp_path)
    assert summary["passed"]
    summary = {"gates": [], "passed": True}
    append_distribution_gate(summary, path)
    assert summary["passed"]
    simulation.write_text("changed simulation inputs", encoding="utf-8")
    summary = {"gates": [], "passed": True}
    append_distribution_gate(summary, path)
    assert not summary["passed"]
    simulation.write_text("simulation inputs", encoding="utf-8")
    production.write_text("changed production inputs", encoding="utf-8")
    summary = {"gates": [], "passed": True}
    append_distribution_gate(summary, path, training_dir=tmp_path)
    assert not summary["passed"]


def test_distribution_report_rejects_nan_sample_counts(make_features):
    from edgeops_ai.training.distributions import build_distribution_report, distribution_passed

    report = build_distribution_report(matrix(make_features), matrix(make_features))
    report["production_rows"] = float("nan")
    assert not distribution_passed(report)
