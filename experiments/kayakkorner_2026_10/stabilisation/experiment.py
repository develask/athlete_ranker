"""Stabilisation experiment (see ../README.md, experiment 4). Runs KayakKorner's own
engine (prediction/elo in the KayakKorner repository), not this repository's.

Needs annos.json ({athlete_id: birth year}, exported from KayakKorner; not tracked)
next to this file. The configurations at the bottom were edited for each run; the
results of every run are in results/.

¿Se estabiliza el rating? K, multiplicador por incertidumbre y λ.

Para cada configuración:
- predicción (log loss y acierto de grupos, sin mirar el futuro);
- salto medio por regata en la distancia corrida, de todos y de los asentados (n ≥ 20);
- deriva anual: rating a final de cada año en cada distancia (años con ≥ 3 carreras en ella);
  cambio entre años consecutivos a partir del 2º año de carrera, por edad.
"""
import itertools, json, statistics, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

KK = Path.home() / "projects/KayakKorner"
sys.path.insert(0, str(KK))
from prediction.elo.grupos import grupos_de_regata
from prediction.elo.motor import Metricas, Parametros, procesar_grupo
from scripts.validar_elo import regatas_del_dataset
sys.path.insert(0, str(Path(__file__).parent))
import starts as arranques

AQUI = Path(__file__).parent
REGATAS = None
ANNOS = None


def init():
    global REGATAS, ANNOS
    datos = json.load(open(Path.home() / "projects/athlete_ranker/data/processed/dataset_kk.json"))
    arranques.META.update({int(p["prueba_id"]): (p.get("categoria"), p.get("sexo"))
                           for r in datos for p in r["pruebas"]})
    REGATAS = sorted(regatas_del_dataset(datos), key=lambda r: (r.fecha, r.id))
    ANNOS = {int(k): v for k, v in json.load(open(AQUI / "annos.json")).items()}


def edad_grupo(palista, año):
    nac = ANNOS.get(palista)
    if nac is None:
        return None
    edad = año - nac
    return "juvenil" if edad < 19 else "adulto" if edad < 35 else "veterano"


def correr(config):
    k, kmax, lam, estrategia, fraccion = config
    arranques.ESTRATEGIA, arranques.FRACCION = estrategia, fraccion
    arranques.reiniciar()
    p = Parametros(k=k, incertidumbre_k_max=kmax, lambda_distancia=lam)
    procesar = arranques.procesar_grupo
    medianas = defaultdict(list)  # año -> elos de fondo de asentados al final
    estados = {}
    agrupado = Metricas()
    saltos, saltos_asentados = [], []
    # (palista, clase) -> {año: [elo_final, carreras]}
    anual = defaultdict(dict)
    for regata in REGATAS:
        grupos, _ = grupos_de_regata(regata)
        for grupo in grupos:
            r = procesar(grupo, estados, p)
            agrupado.sumar(r.agrupado)
            if not r.procesado:
                continue
            for palista, cambio in r.cambios.items():
                salto = abs(cambio.distancias[grupo.clase])
                saltos.append(salto)
                rating = estados[palista].ratings[grupo.clase]
                if rating.n_efectiva >= 20:
                    saltos_asentados.append(salto)
                fila = anual[(palista, grupo.clase)].setdefault(regata.fecha.year, [0.0, 0])
                fila[0] = rating.elo
                fila[1] += 1

    for (palista, clase), años in anual.items():
        if clase != "fondo":
            continue
        for año, (elo, n) in años.items():
            if n >= 3:
                medianas[año].append(elo)
    # Mediana de los activos por distancia y año (para medir la posición relativa).
    activos = defaultdict(list)
    for (palista, clase), años in anual.items():
        for año, (elo, n) in años.items():
            if n >= 3:
                activos[(clase, año)].append(elo)
    mediana = {clave: statistics.median(v) for clave, v in activos.items()}
    err1, err2 = [], []
    for (palista, clase), años in anual.items():
        act = sorted(a for a, (_, n) in años.items() if n >= 3)
        if len(act) >= 3 and act[1] == act[0] + 1 and act[2] == act[0] + 2:
            grupo_edad = edad_grupo(palista, act[0])
            if grupo_edad in ("adulto", "veterano"):
                rel = [años[a][0] - mediana[(clase, a)] for a in act[:3]]
                err1.append(abs(rel[0] - rel[2]))
                err2.append(abs(rel[1] - rel[2]))
    # Diferencia entre distancias de un mismo palista.
    from prediction.elo.motor import CLASES
    separaciones, pares_corr = [], defaultdict(list)
    for estado in estados.values():
        corridas = [c for c in CLASES if estado.ratings[c].n_directas >= 3]
        if len(corridas) >= 2:
            elos = [estado.ratings[c].elo for c in corridas]
            separaciones.append(max(elos) - min(elos))
        for a, b in (("super_sprint", "sprint"), ("sprint", "fondo"), ("fondo", "maraton")):
            if a in corridas and b in corridas:
                pares_corr[f"{a[:5]}-{b[:5]}"].append((estado.ratings[a].elo, estado.ratings[b].elo))
    deriva = defaultdict(list)
    for (palista, clase), años in anual.items():
        activos = sorted(a for a, (_, n) in años.items() if n >= 3)
        for i in range(1, len(activos)):
            a0, a1 = activos[i - 1], activos[i]
            if a1 != a0 + 1:
                continue
            grupo_edad = edad_grupo(palista, a1)
            if grupo_edad is None:
                continue
            tramo = "2º año" if i == 1 else "3º+"
            deriva[(grupo_edad, tramo)].append(años[a1][0] - años[a0][0])

    resumen = {
        "k": k, "kmax": kmax, "lambda": lam, "arranque": f"{estrategia} {fraccion}",
        "mediana_fondo": {año: round(statistics.median(v)) for año, v in sorted(medianas.items())},
        "g_ll": agrupado.log_loss / agrupado.pares, "g_acc": agrupado.aciertos / agrupado.decisivos,
        "salto": statistics.mean(saltos), "salto_asentados": statistics.mean(saltos_asentados),
        "ajuste_1er_año": statistics.mean(err1), "ajuste_2º_año": statistics.mean(err2), "n_ajuste": len(err1),
        "separacion_ejes": statistics.median(separaciones),
        "correlaciones": {k: round(statistics.correlation(*zip(*v)), 2) for k, v in pares_corr.items()},
        "cruzados": {t: {"pares": f[0], "acierto": f[1] / f[2], "log_loss": f[3] / f[0],
                         "predicho": f[4] / f[0], "real": f[5] / f[0]}
                     for t, f in arranques.CRUZADOS.items()},
    }
    for (grupo_edad, tramo), valores in sorted(deriva.items()):
        resumen[f"{grupo_edad} {tramo}"] = (len(valores), statistics.mean(valores), statistics.mean(map(abs, valores)))
    return resumen


if __name__ == "__main__":
    configs = [(60.0, 1.5, 20.0, "base", 0), (60.0, 1.5, 20.0, "rend", 0.3), (40.0, 3.0, 10.0, "rend", 0.3),
               (40.0, 3.0, 20.0, "rend", 0.3), (40.0, 5.0, 5.0, "rend", 0.3), (60.0, 3.0, 5.0, "rend", 0.3)]
    with ProcessPoolExecutor(max_workers=6, initializer=init) as ex:
        resultados = list(ex.map(correr, configs))
    json.dump(resultados, open(AQUI / "resultados_ajuste2.json", "w"), indent=1, default=str)
    for r in resultados:
        c = r["cruzados"]
        print(f"K={r['k']:.0f} kmax={r['kmax']} λ={r['lambda']:.0f} {r['arranque']:9} | ll={r['g_ll']:.4f} acc={r['g_acc']:.3f} "
              f"| salto={r['salto']:4.1f} asent={r['salto_asentados']:4.1f} | ejes={r['separacion_ejes']:.0f} {r['correlaciones']}")
        for t in ("entre asentados", "con novato", "sexo distinto", "categoría distinta"):
            f = c[t]
            print(f"    {t:18} pares={f['pares']:>9} ll={f['log_loss']:.4f} acc={f['acierto']:.3f} "
                  f"predicho={f['predicho']:.3f} real={f['real']:.3f} sesgo={f['predicho'] - f['real']:+.3f}")
