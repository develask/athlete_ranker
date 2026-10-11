"""Anchor tests for the time rating (see ../README.md, experiment 5).

The model estimates the difficulty μ of every prueba (here only with established
boats and without sharing anything with the group, see ModeloTiempos.mus). The
real distance does not matter, only that the course is the same:

1. Mismo recorrido, años distintos: μ = recorrido + año + ruido (efectos fijos
   robustos, medianas alternadas). Recorrido = sede × distancia anunciada × barco ×
   categoría × sexo, en aguas tranquilas, en al menos MIN_AÑOS años. Si la escala no
   deriva, el efecto año es plano salvo el tiempo que hizo. Pendiente < 0: en el mismo
   recorrido se va más rápido de lo que dicen los niveles (los niveles se quedan
   cortos ante la mejora real, o el deporte mejora); > 0, al revés. Intervalo por
   bootstrap de sedes.
2. Mismo día, sede, distancia y barco, categorías distintas: μ de la prueba menos
   μ de la de seniors. Debería ser 0 (mismo agua, misma distancia). > 0: los tiempos
   de esa categoría son más lentos de lo que dicen sus niveles frente a los seniors
   (sus niveles están altos).

lugares.json: {regata_id: lugar} exportado de Eventos.lugar en KayakKorner.
"""
import json, pickle, random, re, statistics, sys, unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import experiment as X

AQUI = Path(__file__).parent
MIN_AÑOS = 4
QUIETAS = ("embalse", "encoro", "pantano", "lago", "lagoa", "canal", "cear", "pista", "presa", "estanque")


def normalizar(lugar):
    t = unicodedata.normalize("NFKD", lugar.lower()).encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\b(embalse|encoro|pantano|lago|lagoa|de|del|da|do|la|el|o|a)\b", " ", t)
    return " ".join(t.split())


def tipo_agua(lugar):
    t = unicodedata.normalize("NFKD", lugar.lower()).encode("ascii", "ignore").decode()
    if any(p in t for p in QUIETAS):
        return "quieta"
    if re.search(r"\brio\b", t):
        return "rio"
    return "otro"


def meta():
    lugares = json.load(open(AQUI / "lugares.json"))
    m = {}
    for r in json.load(open(X.DATOS)):
        lugar = lugares.get(r["regata_id"])
        for p in r["pruebas"]:
            m[int(p["prueba_id"])] = dict(
                regata=int(r["regata_id"]), año=int(r["fecha"][:4]), lugar=lugar,
                sede=normalizar(lugar) if lugar else None, agua=tipo_agua(lugar) if lugar else None,
                distancia=p["distancia_exacta"], barco=(p["embarcacion_tipo"], p["embarcacion_num"]),
                categoria=p.get("categoria"), sexo=p.get("sexo"), tipo=p["tipo"])
    return m


def mus_del_modelo(cache, cats, config):
    modelo = X.ModeloTiempos(**config, cats=cats)
    for g in cache["grupos"]:
        modelo.predecir_y_actualizar(g)
    return modelo.mus


def efectos_fijos(obs, iteraciones=15):
    """obs: [(recorrido, año, μ)]. Devuelve {año: efecto} (medianas alternadas, media 0)."""
    año_ef = defaultdict(float)
    for _ in range(iteraciones):
        por_rec = defaultdict(list)
        for rec, a, mu in obs:
            por_rec[rec].append(mu - año_ef[a])
        rec_ef = {k: statistics.median(v) for k, v in por_rec.items()}
        por_año = defaultdict(list)
        for rec, a, mu in obs:
            por_año[a].append(mu - rec_ef[rec])
        año_ef = defaultdict(float, {a: statistics.median(v) for a, v in por_año.items()})
        media = statistics.mean(año_ef.values())
        año_ef = defaultdict(float, {a: v - media for a, v in año_ef.items()})
    return dict(año_ef)


def pendiente(efectos):
    xs = sorted(efectos)
    if len(xs) < 3:
        return None
    mx = statistics.mean(xs); my = statistics.mean(efectos[x] for x in xs)
    return sum((x - mx) * (efectos[x] - my) for x in xs) / sum((x - mx) ** 2 for x in xs)


def prueba_recorrido(filas, filtro, reps=300):
    filas = [f for f in filas if filtro(f)]
    años_rec = defaultdict(set)
    for f in filas:
        años_rec[f["recorrido"]].add(f["año"])
    filas = [f for f in filas if len(años_rec[f["recorrido"]]) >= MIN_AÑOS]
    if not filas:
        return None
    obs = [(f["recorrido"], f["año"], f["mu"]) for f in filas]
    ef = efectos_fijos(obs)
    por_sede = defaultdict(list)
    for f in filas:
        por_sede[f["sede"]].append((f["recorrido"], f["año"], f["mu"]))
    sedes = sorted(por_sede)
    random.seed(1)
    boot = []
    for _ in range(reps):
        muestra = []
        for k, sede in enumerate(random.choices(sedes, k=len(sedes))):
            muestra += [((k, rec), a, mu) for rec, a, mu in por_sede[sede]]
        b = pendiente(efectos_fijos(muestra, 8))
        if b is not None:
            boot.append(b)
    boot.sort()
    n_año = defaultdict(int)
    for f in filas:
        n_año[f["año"]] += 1
    return {"pruebas": len(filas), "recorridos": len(años_rec and {f["recorrido"] for f in filas}),
            "sedes": len(sedes),
            "efecto_año_pct": {a: round(100 * v, 2) for a, v in sorted(ef.items())},
            "pruebas_por_año": dict(sorted(n_año.items())),
            "pendiente_pct_año": round(100 * pendiente(ef), 3),
            "ic90_pct_año": [round(100 * boot[int(0.05 * len(boot))], 3), round(100 * boot[int(0.95 * len(boot))], 3)]}


def prueba_categorias(filas):
    celdas = defaultdict(list)
    for f in filas:
        if f["tipo"] == "aguas_tranquilas" and f["categoria"]:
            celdas[(f["regata"], f["distancia"], f["barco"])].append(f)
    difs = defaultdict(list)
    for celda in celdas.values():
        seniors = [f["mu"] for f in celda if f["categoria"] == "senior"]
        if not seniors or len(celda) < 2:
            continue
        ref = statistics.median(seniors)
        for f in celda:
            clave = f["categoria"] if f["categoria"] != "senior" else f"senior_{f['sexo']}"
            difs[clave].append(f["mu"] - ref)
            periodo = "2015-2019" if f["año"] <= 2019 else "2020-2026"
            difs[f"{clave} {periodo}"].append(f["mu"] - ref)
            difs[f"{clave} {f['año']}"].append(f["mu"] - ref)
    return {k: {"n": len(v), "mediana_pct": round(100 * statistics.median(v), 2),
                "media_pct": round(100 * statistics.mean(v), 2)}
            for k, v in sorted(difs.items()) if len(v) >= 20}


def main(variantes, parte=None):
    """parte: «ajuste» o «prueba» para mirar solo esa mitad de las sedes (calibrar.mitad)."""
    cache = pickle.load(open(X.CACHE, "rb"))
    cats = X.categorias()
    m = meta()
    salida = {}
    for nombre in variantes:
        filas = []
        for prueba_id, mu, n in mus_del_modelo(cache, cats, X.VARIANTES[nombre]):
            info = m.get(prueba_id)
            if not info:
                continue
            if parte and calibrar_mitad(info["sede"]) != parte:
                continue
            filas.append(dict(info, mu=mu, n=n,
                              recorrido=(info["sede"], info["distancia"], info["barco"], info["categoria"],
                                         info["sexo"])))
        quietas = lambda f: f["tipo"] == "aguas_tranquilas" and f["sede"] and f["distancia"]
        mayores = ("junior", "sub23", "senior", "veterano")
        sin_rios = lambda f: quietas(f) and f["agua"] != "rio" and f["año"] >= 2019
        salida[nombre] = {
            "recorrido": {
                "todas": prueba_recorrido(filas, quietas),
                "agua_quieta": prueba_recorrido(filas, lambda f: quietas(f) and f["agua"] == "quieta"),
                "rio": prueba_recorrido(filas, lambda f: quietas(f) and f["agua"] == "rio"),
                "otras_sedes": prueba_recorrido(filas, lambda f: quietas(f) and f["agua"] == "otro"),
                "junior_y_mayores": prueba_recorrido(filas, lambda f: quietas(f) and f["categoria"] in mayores),
                "menores": prueba_recorrido(filas, lambda f: quietas(f) and f["categoria"] not in mayores),
                "veteranos": prueba_recorrido(filas, lambda f: quietas(f) and f["categoria"] == "veterano"),
                # Sin ríos (el caudal cambia) y desde 2019 (2015–2018 aún arrastran el arranque en frío).
                "junior_y_mayores_2019": prueba_recorrido(filas, lambda f: sin_rios(f) and f["categoria"] in mayores),
                "menores_2019": prueba_recorrido(filas, lambda f: sin_rios(f) and f["categoria"] not in mayores),
                "veteranos_2019": prueba_recorrido(filas, lambda f: sin_rios(f) and f["categoria"] == "veterano"),
            },
            "categorias": prueba_categorias(filas),
        }
        print(nombre, "listo", flush=True)
    json.dump(salida, open(AQUI / f"results/anclas{'_' + parte if parte else ''}.json", "w"), indent=1, default=str)


def calibrar_mitad(sede):
    from calibrar import mitad
    return mitad(sede)


if __name__ == "__main__":
    # python anclas.py [variante ...] [--sedes ajuste|prueba]
    args = sys.argv[1:]
    parte = None
    if "--sedes" in args:
        k = args.index("--sedes"); parte = args[k + 1]; del args[k:k + 2]
    main(args or ["por_prueba", "grupo_tau2_qcat"], parte)
