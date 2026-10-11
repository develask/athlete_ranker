"""Calibrates the children's expected improvement and starting level of the time rating
(see ../README.md, experiment 5) with the anchor tests of anclas.py.

Rejilla de (mejora, arranque) sobre la variante grupo_tau2_qcat. Las sedes se reparten
en dos mitades fijas (crc32 del nombre normalizado): con la de ajuste se elige la
pareja que deja a cero, para los menores, la pendiente en el mismo recorrido (prueba 1)
y el desfase frente a los seniors el mismo día (prueba 2, cadete, infantil y junior);
la otra mitad solo se mira al final (anclas.py con --sedes prueba).

    python calibrar.py                         # rejilla → results/calibracion.jsonl
    python calibrar.py 0.11,0.055 0.12,0.06    # solo esas parejas
"""
import json, pickle, statistics, sys, zlib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import anclas as A
import experiment as X

AQUI = Path(__file__).parent
MAYORES = ("junior", "sub23", "senior", "veterano")
MENORES_DESFASE = ("infantil", "cadete", "junior")


def mitad(sede):
    return "ajuste" if zlib.crc32((sede or "").encode()) % 2 == 0 else "prueba"


def filas_de(cache, cats, meta, config):
    filas = []
    for pid, mu, n in A.mus_del_modelo(cache, cats, config):
        info = meta.get(pid)
        if info:
            filas.append(dict(info, mu=mu, recorrido=(info["sede"], info["distancia"], info["barco"],
                                                      info["categoria"], info["sexo"])))
    return filas


def pendiente_recorrido(filas, filtro):
    """Pendiente (en %/año) del efecto año con efectos fijos de recorrido, sin bootstrap."""
    filas = [f for f in filas if filtro(f)]
    años = defaultdict(set)
    for f in filas:
        años[f["recorrido"]].add(f["año"])
    obs = [(f["recorrido"], f["año"], f["mu"]) for f in filas if len(años[f["recorrido"]]) >= A.MIN_AÑOS]
    if len(obs) < 50:
        return None, len(obs)
    return 100 * A.pendiente(A.efectos_fijos(obs)), len(obs)


def desfases(filas, filtro):
    """{categoría: mediana de μ − μ de los seniors en la misma regata, distancia y barco} (%)."""
    celdas = defaultdict(list)
    for f in filas:
        if filtro(f) and f["categoria"]:
            celdas[(f["regata"], f["distancia"], f["barco"])].append(f)
    difs = defaultdict(list)
    for celda in celdas.values():
        seniors = [f["mu"] for f in celda if f["categoria"] == "senior"]
        if seniors:
            ref = statistics.median(seniors)
            for f in celda:
                difs[f["categoria"]].append(f["mu"] - ref)
    return {c: (100 * statistics.median(v), len(v)) for c, v in difs.items() if len(v) >= 15}


def evaluar(filas, parte):
    base = lambda f: (f["tipo"] == "aguas_tranquilas" and f["sede"] and f["distancia"] and f["agua"] != "rio"
                      and f["año"] >= 2019 and (parte is None or mitad(f["sede"]) == parte))
    menores, n_men = pendiente_recorrido(filas, lambda f: base(f) and f["categoria"] not in MAYORES)
    mayores, n_may = pendiente_recorrido(filas, lambda f: base(f) and f["categoria"] in MAYORES)
    d = desfases(filas, base)
    objetivo = (menores or 0) ** 2 + statistics.mean(d[c][0] ** 2 for c in MENORES_DESFASE if c in d)
    return {"pendiente_menores": menores, "n_menores": n_men, "pendiente_mayores": mayores, "n_mayores": n_may,
            "desfases": d, "objetivo": objetivo}


def main():
    cache = pickle.load(open(X.CACHE, "rb"))
    cats = X.categorias()
    meta = A.meta()
    parejas = [tuple(map(float, a.split(","))) for a in sys.argv[1:]] or \
        [(m, a) for m in (0.0, 0.03, 0.06, 0.09, 0.12) for a in (0.0, 0.01, 0.02, 0.03, 0.04)]
    filas_res = []
    for mejora, arranque in parejas:
        cfg = dict(X.VARIANTES["grupo_tau2_qcat"], mejora=mejora, arranque=arranque)
        r = evaluar(filas_de(cache, cats, meta, cfg), "ajuste")
        filas_res.append({"mejora": mejora, "arranque": arranque, **r})
        d = r["desfases"]
        print(f"mejora {mejora:.3f} arranque {arranque:.3f} | pendiente menores {r['pendiente_menores']:+.2f} "
              f"mayores {r['pendiente_mayores']:+.2f} | " +
              " ".join(f"{c} {d[c][0]:+.1f}" for c in ("infantil", "cadete", "junior", "veterano") if c in d) +
              f" | obj {r['objetivo']:.2f}", flush=True)
    return filas_res


if __name__ == "__main__":
    # Con parejas «mejora,arranque» como argumentos, solo esas (para lanzar varias en paralelo);
    # el resultado de cada una se añade a results/calibracion.jsonl.
    with open(AQUI / "results/calibracion.jsonl", "a") as salida:
        for fila in main():
            salida.write(json.dumps(fila) + "\n")
