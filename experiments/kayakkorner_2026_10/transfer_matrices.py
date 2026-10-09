"""Option A fixed; only the length transfer matrix changes. Measures prediction and how
different each athlete's four distance ratings end up (what a radar chart shows).

Usage (from elo/): python ../experiments/kayakkorner_2026_10/transfer_matrices.py OUT.json
"""
import itertools, json, statistics, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
ELO = Path(__file__).resolve().parents[2] / "elo"
DATASET = ELO.parent / "data" / "processed" / "dataset_kk.json"  # KayakKorner: scripts/export_dataset_elo.py
sys.path.insert(0, str(ELO))
import rating_engine
import run_elo_v1 as r
from performance_groups import prepare_performance_groups

L = rating_engine.LENGTH_CLASSES
def matriz(vecina, sprint_fondo=0.0, lejana=0.0, muy_lejana=0.0):
    m = {a: {b: 0.0 for b in L} for a in L}
    for a in L: m[a][a] = 1.0
    for a, b, v in [("super_sprint", "sprint", vecina), ("fondo", "maraton", vecina), ("sprint", "fondo", sprint_fondo),
                    ("super_sprint", "fondo", lejana), ("sprint", "maraton", lejana), ("super_sprint", "maraton", muy_lejana)]:
        m[a][b] = m[b][a] = v
    return m
G = r.HYPERPARAM_GRID["length_transfer"]
MATRICES = {
    "0 (nada)": G[0], "7 (actual)": rating_engine.LENGTH_TRANSFER, "V25": matriz(0.25, 0.05), "2": G[2],
    "V50": matriz(0.5), "3": G[3], "4": G[4],
}
A = dict(k=75.0, field_size_reference=20.0, field_size_alpha=0.5, field_size_min_multiplier=0.7, field_size_max_multiplier=1.5,
         context_mode="modifiers", team_update_mode="inverse_sqrt", inactivity_mode="evidence_decay",
         inactivity_grace_days=365.0, evidence_half_life_days=180.0, uncertainty_k_max_multiplier=1.0,
         primary_switch_delta=999_999_999)
GRUPOS = None
def init(g):
    global GRUPOS; GRUPOS = g

def correr(nombre):
    rating_engine.PRIMARY_LAMBDA, rating_engine.MODIFIER_LAMBDA = 15.0, 5.0
    rating_engine.LENGTH_TRANSFER = rating_engine.CONFIDENCE_TRANSFER = MATRICES[nombre]
    tracker = r.PopulationHealthTracker("modifiers")
    atletas, _, ev, _ = rating_engine.run_rating_system(GRUPOS, evaluation_groups=GRUPOS, population_tracker=tracker, **A)
    sin, con = r.evaluation_summary(ev["ungrouped"]), r.evaluation_summary(ev["grouped"])
    pob = tracker.aggregate()
    # Diferencias entre las distancias de un mismo palista (las que ha corrido de verdad, ≥3 veces)
    pares = {p: [] for p in itertools.combinations(L, 2)}
    rangos = []
    for a in atletas.values():
        corridas = [c for c in L if a.length_ratings[c].n_direct >= 3]
        for x, y in itertools.combinations(corridas, 2):
            pares[(x, y)].append((a.length_ratings[x].elo, a.length_ratings[y].elo))
        if len(corridas) >= 2:
            elos = [a.length_ratings[c].elo for c in corridas]
            rangos.append(max(elos) - min(elos))
    def corr(v):
        if len(v) < 30: return None
        xs, ys = zip(*v); return statistics.correlation(xs, ys)
    std_pob = pob["final_rating_std"] if "final_rating_std" in pob else None
    return {
        "matriz": nombre, "ll": sin["log_loss"], "br": sin["brier_score"], "acc": sin["decisive_accuracy"],
        "g_ll": con["log_loss"], "g_br": con["brier_score"], "g_acc": con["decisive_accuracy"],
        "salto": pob["absolute_updates"]["mean"], "std": std_pob,
        "palistas_multi": len(rangos), "rango_medio": statistics.mean(rangos), "rango_mediana": statistics.median(rangos),
        "pares": {f"{x[:5]}-{y[:5]}": (len(v), statistics.mean(abs(a - b) for a, b in v) if v else None, corr(v))
                  for (x, y), v in pares.items()},
    }

if __name__ == "__main__":
    datos = json.load(open(DATASET))
    grupos, _ = prepare_performance_groups(datos, keep_unusable=True, group_pruebas=True)
    with ProcessPoolExecutor(max_workers=7, initializer=init, initargs=(grupos,)) as ex:
        res = list(ex.map(correr, MATRICES))
    json.dump(res, open(sys.argv[1], "w"), indent=1, default=str)
    for x in res:
        print(f"{x['matriz']:11} g_ll={x['g_ll']:.4f} g_acc={x['g_acc']:.4f} ll={x['ll']:.4f} salto={x['salto']:.1f} std={x['std']} "
              f"| multi={x['palistas_multi']} rango={x['rango_medio']:.0f}/{x['rango_mediana']:.0f} | "
              + " ".join(f"{k}:{v[1]:.0f}/r{v[2]:.2f}" for k, v in x['pares'].items() if v[1] is not None and v[2] is not None))
