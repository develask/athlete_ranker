"""Fine search around option A: grouped pruebas, context modifiers, length transfer matrix 2.

Usage (from elo/): python ../experiments/kayakkorner_2026_10/search_fine.py OUT_DIR N_EXPERIMENTS N_JOBS
"""
import csv, json, sys
from pathlib import Path
ELO = Path(__file__).resolve().parents[2] / "elo"
DATASET = ELO.parent / "data" / "processed" / "dataset_kk.json"  # KayakKorner: scripts/export_dataset_elo.py
sys.path.insert(0, str(ELO))
import run_elo_v1 as r
from performance_groups import prepare_performance_groups

GRID = {
    "group_pruebas": (True,),
    "context_mode": ("modifiers",),
    "team_update_mode": ("inverse_sqrt", "none"),
    "inactivity_mode": ("evidence_decay", "none"),
    "inactivity_grace_days": (180.0, 365.0),
    "evidence_half_life_days": (180.0, 365.0, 730.0),
    "uncertainty_k_max_multiplier": (1.0, 1.5, 2.0),
    "primary_switch_delta": (5, 10, 20, 999_999_999),
    "k_factor": (40.0, 50.0, 60.0, 75.0, 90.0, 100.0),
    "field_size_alpha": (0.25, 0.5, 0.75, 1.0),
    "field_size_reference": (10.0, 20.0, 30.0, 50.0),
    "field_size_min_multiplier": (0.5, 0.7, 0.8),
    "field_size_max_multiplier": (1.25, 1.5, 1.75, 2.0),
    "primary_lambda": (10.0, 15.0, 20.0),
    "modifier_lambda": (5.0, 10.0),
    "length_transfer": (r.HYPERPARAM_GRID["length_transfer"][2],),
}

def main():
    salida = Path(sys.argv[1]); n = int(sys.argv[2]); jobs = int(sys.argv[3])
    salida.mkdir(parents=True, exist_ok=True)
    datos = json.load(open(DATASET))
    grupos = {True: prepare_performance_groups(datos, keep_unusable=True, group_pruebas=True)[0]}
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
