# -*- coding: utf-8 -*-
"""The unattended runner must be resumable and must not fabricate results."""
import importlib.util
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_newtrain",
        pathlib.Path(__file__).resolve().parent.parent / "scripts" / "run_newtrain.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_matrix_is_the_two_specified_runs():
    mod = _load()
    names = [r["run_name"] for r in mod.RUNS]
    assert names == ["dcs_actuator_geom", "dcs_actuator_geom_nope"]
    a, b = mod.RUNS
    assert a["config"].endswith("dcs_model_geom.yml")      # with time PE
    assert b["config"].endswith("dcs_model.yml")           # PE ablation
    assert all(r["npz_dir"].endswith("NpzGeom") for r in mod.RUNS)


def test_done_detects_a_finished_unit(tmp_path):
    mod = _load()
    run_dir = tmp_path / "dcs_actuator_geom"
    run_dir.mkdir()
    assert not mod.is_done(run_dir, "m0")
    (run_dir / "m0.joblib").write_text("x")
    assert not mod.is_done(run_dir, "m0"), "an artifact alone is not a finished unit"
    (run_dir / "m0_pred.npz").write_text("x")
    assert mod.is_done(run_dir, "m0")


def test_report_marks_missing_rows_rather_than_inventing_them(tmp_path):
    mod = _load()
    csv_path = tmp_path / "bench.csv"
    csv_path.write_text(
        "run,model,ccc,r2,similarity,rmse_cm,ccc_p90,n_shots,rgeom_mae_mm,"
        "zgeom_mae_mm,centre_rmse_mm,abs_bnd_rmse_mm\n"
        "dcs_actuator_geom,m0,0.96,0.85,0.999,2.0,0.98,76,1.5,0.9,1.8,3.1\n")
    out = tmp_path / "report.md"
    mod.write_report(csv_path, out)
    text = out.read_text()
    assert "dcs_actuator_geom" in text and "0.96" in text
    assert "not run" in text, "absent runs must be shown as absent"
    assert "0.9579" in text, "the baseline must appear for comparison"
