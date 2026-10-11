"""KayakKorner's radar and evolution charts for one or more athletes, with the
current Elo and the time rating side by side (see ../README.md, experiment 5).

    python graficas.py                        # servidor en http://localhost:8765 con buscador por nombre
    python graficas.py 11700 11905            # HTML estático, por id
    python graficas.py "garcia lopez"         # por nombre (lista coincidencias si hay varias)
    python graficas.py 11700 --salida /tmp/a.html

Escribe un HTML autónomo (por defecto results/graficas.html). El Elo de cada palista
sale de EloHistorico de la base local de KayakKorner (lo que ve la web); el modelo
de tiempos se recalcula (variante VARIANTE, unos 20 s).

Escala del modelo de tiempos: puntos con significado fijo, 1500 = la mediana de los
activos (≥ 3 carreras en la distancia) en AÑO_BASE, todas las categorías y sexos, en cada
distancia; PUNTOS_POR_PCT puntos = 1 % de velocidad. No cambia con los años: 1600 en 2019
y en 2026 son el mismo nivel.

Radar: la misma geometría que la web (palista-radar/radar-geometria.ts): cuatro
ejes, respecto a la categoría del elegido, radio = (valor − p50) / ((p90 − p10) / 2,563)
con escala fija ±3, el rombo central es la mediana. Los cortes se calculan igual para
los dos sistemas: activos del último año con ≥ 3 carreras en la distancia, por
categoría (la más corrida ese año) y sexo. Margen: el de la web para el Elo
(35 + 60·incertidumbre); una desviación típica para el modelo de tiempos.

Evolución: como comparison-elo-history: una línea por palista en la distancia
elegida, la franja del margen y la mediana de su categoría (por año) del elegido.
"""
import argparse, json, math, os, pickle, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import experiment as X

KK = X.KK
AQUI = Path(__file__).parent
CLASES = X.CLASES
ETIQUETAS = {"super_sprint": "Super sprint", "sprint": "Sprint", "fondo": "Fondo", "maraton": "Maratón"}
LAMBDA = X.Parametros().lambda_distancia
VARIANTE = "grupo_tau2_qcat_mejora"
AÑO_BASE = 2025
PUNTOS_POR_PCT = 25.0


def conexion():
    import psycopg
    from dotenv import load_dotenv
    load_dotenv(KK / ".env"); load_dotenv(KK / ".env.local", override=True)
    return psycopg.connect(
        f"host={os.environ.get('DATABASE_HOST', 'localhost')} port={os.environ.get('DATABASE_PORT', 5432)} "
        f"user={os.environ['DATABASE_USER']} password={os.environ['DATABASE_PASSWORD']} "
        f"dbname={os.environ['DATABASE_NAME']}")


def resolver(cur, argumentos):
    ids = []
    for a in argumentos:
        if a.isdigit():
            ids.append(int(a))
            continue
        palabras = a.lower().split()
        cur.execute("SELECT id, nombre, apellido FROM Palistas WHERE " +
                    " AND ".join(["lower(nombre || ' ' || apellido) LIKE %s"] * len(palabras)) + " ORDER BY id",
                    [f"%{p}%" for p in palabras])
        filas = cur.fetchall()
        if len(filas) != 1:
            print(f"«{a}»: {len(filas)} coincidencias" + (", usa el id:" if filas else ""))
            for f in filas[:30]:
                print(f"  {f[0]}  {f[1]} {f[2]}")
            sys.exit(1)
        ids.append(filas[0][0])
    return ids


def nombres(cur, ids):
    cur.execute("SELECT id, nombre, apellido FROM Palistas WHERE id = ANY(%s)", [ids])
    return {i: f"{n} {a}".strip().lower() for i, n, a in cur.fetchall()}


def elo_trayectorias(cur, ids):
    """{palista: {clase: [(fecha, regata, elo, margen)]}} desde EloHistorico."""
    cur.execute("""
        SELECT palista, fecha, regata, elo_super_sprint, elo_sprint, elo_fondo, elo_maraton,
               ef_super_sprint, ef_sprint, ef_fondo, ef_maraton
        FROM EloHistorico WHERE palista = ANY(%s) ORDER BY palista, fecha, regata""", [ids])
    out = defaultdict(lambda: defaultdict(list))
    for palista, fecha, regata, *v in cur.fetchall():
        for k, clase in enumerate(CLASES):
            margen = 35 + 60 * LAMBDA / (LAMBDA + max(0.0, v[4 + k]))
            out[palista][clase].append((fecha.isoformat(), regata, v[k], margen))
    return out


def perfil_categoria():
    """{(palista, año): (categoría, sexo)}: la categoría más corrida ese año y su sexo más corrido."""
    cat = defaultdict(Counter); sexo = defaultdict(Counter)
    for r in json.load(open(X.DATOS)):
        año = int(r["fecha"][:4])
        for p in r["pruebas"]:
            for res in p["results"]:
                for x in res["lista_palista_id"]:
                    if x:
                        if p.get("categoria"):
                            cat[(int(x), año)][p["categoria"]] += 1
                        if p.get("sexo") in ("masculino", "femenino"):
                            sexo[int(x)][p["sexo"]] += 1
    return {k: (v.most_common(1)[0][0], sexo[k[0]].most_common(1)[0][0] if sexo[k[0]] else None)
            for k, v in cat.items()}


def cortes(valores):
    v = sorted(valores)
    q = lambda p: v[min(len(v) - 1, int(p * len(v)))]
    return [q(.10), q(.25), q(.50), q(.75), q(.90)]


def bandas(finales, perfil, convertir=lambda clase, v: v):
    """{(clase, año, categoría, sexo): cortes} de los activos (≥ 3 carreras) de cada año."""
    grupos = defaultdict(list)
    for (palista, clase), años in finales.items():
        for año, (valor, n) in años.items():
            if n >= 3 and (palista, año) in perfil:
                grupos[(clase, año) + perfil[(palista, año)]].append(convertir(clase, valor))
    return {k: cortes(v) for k, v in grupos.items() if len(v) >= 10}


def tiempos(cache, ids):
    """Trayectorias del modelo de tiempos para `ids` (None: todos) y los finales de año de todos."""
    cats = X.categorias()
    modelo = X.ModeloTiempos(**X.VARIANTES[VARIANTE], cats=cats)
    modelo.registrar = True
    quiero = None if ids is None else set(ids)
    tray = defaultdict(lambda: defaultdict(list))
    finales = defaultdict(dict)
    for g in cache["grupos"]:
        _, estados = modelo.predecir_y_actualizar(g)
        año = g["fecha"].year
        for (tripulacion, _), ss in zip(g["botes"], estados):
            for pal, s in zip(tripulacion, ss):
                fila = finales[(pal, g["clase"])].setdefault(año, [0.0, 0])
                fila[0] = s[0]; fila[1] += 1
                if quiero is None or pal in quiero:
                    tray[pal][g["clase"]].append((g["fecha"].isoformat(), g["regata"], 100 * s[0],
                                                  100 * s[1] ** 0.5))
    return tray, finales, modelo


def meta_pruebas():
    """{prueba: datos de la prueba} y {(prueba, palista): (puesto, botes)} del dataset."""
    meta, puestos = {}, {}
    for r in json.load(open(X.DATOS)):
        for p in r["pruebas"]:
            pid = int(p["prueba_id"])
            meta[pid] = {"fecha": r["fecha"], "regata": r["regata_name"], "prueba": p["prueba_name"],
                         "distancia": p["distancia_exacta"], "barco": (p["embarcacion_tipo"], p["embarcacion_num"]),
                         "clase": p["length_class"]}
            n = len(p["results"])
            for res in p["results"]:
                for x in res["lista_palista_id"]:
                    if x:
                        puestos[(pid, int(x))] = (res.get("posicion"), n)
    return meta, puestos


def dificultades(pruebas, meta):
    """{prueba: (dificultad, distancia efectiva)}: μ frente a la mediana de las pruebas de la misma
    distancia anunciada y barco (≥ 10 pruebas); la distancia efectiva es la anunciada × e^(μ − típica)."""
    por_tipo = defaultdict(list)
    for pid, (mu, _, _) in pruebas.items():
        m = meta.get(pid)
        if m and m["distancia"]:
            por_tipo[(m["distancia"], m["barco"])].append(mu)
    tipica = {k: statistics.median(v) for k, v in por_tipo.items() if len(v) >= 10}
    out = {}
    for pid, (mu, _, _) in pruebas.items():
        m = meta.get(pid)
        t = tipica.get((m["distancia"], m["barco"])) if m and m["distancia"] else None
        if t is not None:
            d = mu - t
            out[pid] = (d, m["distancia"] * math.exp(d))
    return out


def ultimo(tray):
    return tray[-1] if tray else None


class Contexto:
    """Lo que no depende de los palistas elegidos: el modelo de tiempos (de todos), los perfiles
    de categoría y los cortes de cada sistema."""

    def __init__(self, ids=None):
        cache = pickle.load(open(X.CACHE, "rb"))
        self.perfil = perfil_categoria()
        self.cats_palista = defaultdict(dict)
        for (p, a), v in self.perfil.items():
            self.cats_palista[p][a] = list(v)
        self.tray_t, finales_t, modelo = tiempos(cache, ids)
        self.meta, self.puestos = meta_pruebas()
        self.pruebas_modelo = modelo.pruebas
        self.actuaciones = modelo.actuaciones
        self.dificultad = dificultades(modelo.pruebas, self.meta)
        # La escala: la mediana de los activos del año base en cada distancia vale 1500.
        self.base = {c: statistics.median(años[AÑO_BASE][0] for (p, cl), años in finales_t.items()
                                          if cl == c and años.get(AÑO_BASE, (0, 0))[1] >= 3) for c in CLASES}
        for por_clase in self.tray_t.values():
            for c, puntos in por_clase.items():
                por_clase[c] = [(f, r, self.puntos(c, v / 100), PUNTOS_POR_PCT * sd) for f, r, v, sd in puntos]
        self.bandas = {"elo": bandas(cache["finales"], self.perfil), "tiempos": bandas(finales_t, self.perfil, self.puntos)}
        self.medianas = {}
        for sis, bs in self.bandas.items():
            med = defaultdict(dict)
            for (c, a, cat, sexo), corte in bs.items():
                med[f"{c}|{cat}|{sexo}"][a] = corte[2]
            self.medianas[sis] = med

    def puntos(self, clase, theta):
        return 1500 + PUNTOS_POR_PCT * 100 * (theta - self.base[clase])

    def datos(self, cur, ids):
        nombre = nombres(cur, ids)
        elo = elo_trayectorias(cur, ids)
        tray_t = self.tray_t
        # En la evolución del Elo, solo las regatas en las que corrió esa distancia (EloHistorico guarda
        # las cuatro en cada regata del palista; las otras solo cambian por transferencia).
        for i in ids:
            for c in CLASES:
                directas = {p[1] for p in tray_t.get(i, {}).get(c, [])}
                elo[i][c] = [p for p in elo[i][c] if p[1] in directas]
        trays = {"elo": elo, "tiempos": tray_t}
        datos = {"sistemas": {}, "clases": [{"clave": c, "etiqueta": ETIQUETAS[c]} for c in CLASES],
                 "palistas": [{"id": i, "nombre": nombre.get(i, f"palista {i}"),
                               "categorias_año": self.cats_palista.get(i, {})} for i in ids]}
        for sis in ("elo", "tiempos"):
            perfiles, evol = [], {}
            for i in ids:
                t_i = trays[sis].get(i, {})
                años = sorted({int(p[0][:4]) for c in CLASES for p in t_i.get(c, [])})
                año = años[-1] if años else None
                cat_sexo = self.perfil.get((i, año)) if año else None
                mods = []
                for c in CLASES:
                    t = t_i.get(c, [])
                    u = t[-1] if t else None
                    # Carreras directas: las del modelo de tiempos, que solo guarda esas.
                    directas = len(tray_t.get(i, {}).get(c, []))
                    corte = None
                    if cat_sexo:
                        for a in (año, año - 1):
                            corte = self.bandas[sis].get((c, a) + cat_sexo)
                            if corte:
                                break
                    mods.append({"clave": c, "etiqueta": ETIQUETAS[c], "rating": u[2] if u else None,
                                 "margen": u[3] if u else None, "carreras": directas if u else 0, "cortes": corte})
                perfiles.append({"palista_id": i, "categoria": cat_sexo, "año": año, "modalidades": mods})
                evol[i] = {c: [[p[0], p[2], p[3], p[1]] for p in t_i.get(c, [])] for c in CLASES}
            datos["sistemas"][sis] = {"perfiles": perfiles, "evolucion": evol, "medianas": self.medianas[sis]}
        datos["pruebas"] = {i: self.pruebas_de(i) for i in ids}
        return datos

    def pruebas_de(self, palista):
        """Cada prueba del palista: dificultad, distancia efectiva, nivel del campo, puesto y rendimiento."""
        filas = []
        for pid, e in self.actuaciones.get(palista, []):
            m = self.meta.get(pid)
            if not m:
                continue
            mu, campo, n = self.pruebas_modelo[pid]
            dif = self.dificultad.get(pid)
            puesto = self.puestos.get((pid, palista), (None, n))
            filas.append({"fecha": m["fecha"], "regata": m["regata"], "prueba": m["prueba"], "clase": m["clase"],
                          "distancia": m["distancia"], "puesto": puesto[0], "botes": puesto[1],
                          "dificultad": round(100 * dif[0], 1) if dif else None,
                          "distancia_efectiva": round(dif[1]) if dif else None,
                          "campo": round(self.puntos(m["clase"], campo)) if m["clase"] in self.base else None,
                          "rendimiento": round(-100 * e, 1)})
        return filas


def buscar(cur, texto, perfil_ultimo, limite=20):
    """Palistas con ELO cuyo nombre contiene todas las palabras de `texto`."""
    import unicodedata
    palabras = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode().split()
    if not palabras:
        return []
    cur.execute("SELECT p.id, p.nombre, p.apellido FROM Palistas p WHERE " +
                " AND ".join(["p.nombre_normalizado LIKE %s"] * len(palabras)) +
                " AND EXISTS (SELECT 1 FROM EloHistorico h WHERE h.palista = p.id) ORDER BY p.apellido, p.nombre LIMIT %s",
                [f"%{w}%" for w in palabras] + [limite])
    out = []
    for i, n, a in cur.fetchall():
        ult = perfil_ultimo.get(i)
        out.append({"id": i, "nombre": f"{n} {a}".strip().lower(),
                    "detalle": f"{ult[1][0]} · {ult[0]}" if ult else ""})
    return out


def servir(puerto):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse
    print("calculando el modelo de tiempos de todos…", flush=True)
    ctx = Contexto()
    perfil_ultimo = {}
    for p, años in ctx.cats_palista.items():
        a = max(años)
        perfil_ultimo[p] = (a, años[a])
    plantilla = (AQUI / "graficas_plantilla.html").read_text()
    cx = conexion()

    class Manejador(BaseHTTPRequestHandler):
        def _enviar(self, cuerpo, tipo):
            datos = cuerpo.encode()
            self.send_response(200)
            self.send_header("Content-Type", f"{tipo}; charset=utf-8")
            self.send_header("Content-Length", str(len(datos)))
            self.end_headers()
            self.wfile.write(datos)

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            try:
                if url.path == "/":
                    self._enviar(plantilla, "text/html")
                elif url.path == "/buscar":
                    with cx.cursor() as cur:
                        self._enviar(json.dumps(buscar(cur, q.get("q", [""])[0], perfil_ultimo)), "application/json")
                elif url.path == "/datos":
                    ids = [int(x) for x in q.get("ids", [""])[0].split(",") if x.strip().isdigit()][:6]
                    with cx.cursor() as cur:
                        self._enviar(json.dumps(ctx.datos(cur, ids), default=str), "application/json")
                else:
                    self.send_error(404)
            except Exception as e:  # que un fallo no tumbe el servidor
                cx.rollback()
                self.send_error(500, str(e))

        def log_message(self, *a):
            pass

    print(f"listo: http://localhost:{puerto}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", puerto), Manejador).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("palistas", nargs="*", help="ids o nombres (sin ninguno: servidor con buscador)")
    ap.add_argument("--salida", default=str(AQUI / "results/graficas.html"))
    ap.add_argument("--puerto", type=int, default=8765)
    args = ap.parse_args()
    if not args.palistas:
        servir(args.puerto)
        return
    with conexion() as cx, cx.cursor() as cur:
        ids = resolver(cur, args.palistas)
        datos = Contexto(ids).datos(cur, ids)
    html = (AQUI / "graficas_plantilla.html").read_text().replace("/*DATOS*/null", json.dumps(datos, default=str))
    Path(args.salida).write_text(html)
    print("escrito", args.salida)


if __name__ == "__main__":
    main()
