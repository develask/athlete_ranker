"""Broad random search over the athlete_ranker grid, extended with the current defaults.

Usage (from elo/): python ../experiments/kayakkorner_2026_10/search_broad.py OUT_DIR N_EXPERIMENTS N_JOBS
"""
import csv, json, sys, os
from pathlib import Path
ELO = Path(__file__).resolve().parents[2] / "elo"
DATASET = ELO.parent / "data" / "processed" / "dataset_kk.json"  # KayakKorner: scripts/export_dataset_elo.py
sys.path.insert(0, str(ELO))
import rating_engine
import run_elo_v1 as r
from performance_groups import prepare_performance_groups

GRID = dict(r.HYPERPARAM_GRID)
GRID.update({
    "k_factor": (20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0),
    "evidence_half_life_days": (180.0, 365.0, 730.0),
    "field_size_min_multiplier": (0.5, 0.667, 0.7, 0.8),
    "field_size_max_multiplier": (1.25, 1.5, 1.75, 2.0),
    "length_transfer": tuple(r.HYPERPARAM_GRID["length_transfer"]) + (rating_engine.LENGTH_TRANSFER,),
})

def main():
    salida = Path(sys.argv[1]); n = int(sys.argv[2]); jobs = int(sys.argv[3])
    salida.mkdir(parents=True, exist_ok=True)
    datos = json.load(open(DATASET))
    grupos = {g: prepare_performance_groups(datos, keep_unusable=True, group_pruebas=g)[0] for g in (True, False)}
    resultados = r.hyperparam_search(grupos, param_grid=GRID, n_jobs=jobs, n_experiments=n)
    for res in resultados:
        res.pop("length_transfer", None)
    json.dump(resultados, open(salida / "busqueda.json", "w"), ensure_ascii=False, default=str)
    escalares = [k for k, v in resultados[0].items() if isinstance(v, (int, float, str, bool)) or v is None]
    with open(salida / "busqueda.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=escalares, extrasaction="ignore")
        w.writeheader(); w.writerows(resultados)
    print("hecho", len(resultados))

if __name__ == "__main__":
    main()
