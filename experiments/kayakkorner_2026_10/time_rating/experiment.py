"""Time rating experiment (see ../README.md, experiment 5). Compares KayakKorner's
Elo (prediction/elo in the KayakKorner repository, current parameters) with a
rating in units of time, looking for a scale that is comparable year to year.

Needs data/processed/dataset_kk.json. The first run writes cache.pkl (not tracked).

Modelo de tiempos (filtro de Kalman, un nivel por palista y distancia):

    log(tiempo del bote b en la prueba j) = μ_j − θ_b + ruido(σ_clase)

- θ_b: media de los niveles θ de la tripulación. θ es log-velocidad: +0.01 ≈ 1 % más rápido.
- μ_j: dificultad de la prueba (distancia real, condiciones, recorrido, tipo de barco),
  estimada en la propia prueba con los niveles previos (media ponderada robusta).
  Como todos los botes de la prueba comparten μ_j, el error de la distancia anunciada
  se cancela.
- Cada palista tiene θ y su varianza P. P crece con el tiempo sin correr (paseo
  aleatorio, q por día, hasta P0); un palista nuevo arranca en θ=0 con P0 (o en su
  nivel de otra distancia con más varianza).
- Actualización Kalman con residuos recortados (Huber) para vuelcos, paquetes en
  salidas en grupo y tiempos mal tomados.

Se evalúa como el Elo «dentro de la prueba»: cada par de botes de una prueba, antes de
actualizar. P(a gana) = Φ((θ_a − θ_b) / √(V_a + V_b + 2σ²)).

Pasada 1 (lenta, una vez): reproduce el Elo y guarda en cache.pkl, por grupo, el
rating de cada bote con el que predijo y los ratings de final de año.
Pasada 2: el modelo de tiempos con cada configuración y las métricas de los dos.

Métricas:
- predicción sobre los mismos pares: log loss, acierto, Brier;
- escala s por año: el s que mejor ajusta P(d·s) (s=1 calibrado; si s cambia entre
  años, una misma diferencia no significa lo mismo);
- deriva anual por categoría (la más corrida ese año; los años de nacimiento de
  KayakKorner son inferidos de la categoría y no separan adultos de veteranos): cambio
  del rating entre temporadas seguidas (≥ 3 carreras en la distancia en las dos), desde
  la 3.ª temporada en esa distancia;
- dispersión: desviación típica del nivel de los asentados activos cada año.
"""
import json, math, pickle, random, statistics, sys
from collections import defaultdict
from pathlib import Path

KK = Path.home() / "projects/KayakKorner"
sys.path.insert(0, str(KK))

AQUI = Path(__file__).parent
DATOS = Path.home() / "projects/athlete_ranker/data/processed/dataset_kk.json"
CACHE = AQUI / "cache.pkl"
CLASES = ("super_sprint", "sprint", "fondo", "maraton")
LN10 = math.log(10)


def esperado(a, b):
    """La probabilidad del Elo de que gane a (la de prediction/elo/motor.py)."""
    return 1.0 / (1.0 + 10.0 ** ((b - a) / 400.0))


# ---------------------------------------------------------------------------
# Pasada 1: el Elo
# ---------------------------------------------------------------------------

def unidades(grupo):
    """Las pruebas del grupo (entradas de evaluación), cada una como lista de índices."""
    ev = grupo.entradas_evaluacion
    return [[i for i, e in enumerate(ev) if e.indice_prueba == k] for k in sorted({e.indice_prueba for e in ev})]


def pasada_elo():
    """Reproduce el ELO de KayakKorner. KayakKorner lo sustituyó por el Índice de
    rendimiento (rama feat/indice-rendimiento): para regenerar cache.pkl hace falta
    una versión anterior, con prediction/elo (por ejemplo el commit e044025)."""
    from prediction.elo import motor
    from prediction.elo.grupos import grupos_de_regata
    from prediction.elo.motor import Parametros
    from scripts.validar_elo import regatas_del_dataset
    datos = json.load(open(DATOS))
    regatas = sorted(regatas_del_dataset(datos), key=lambda r: (r.fecha, r.id))
    capturado = {}
    original = motor._evaluar

    def evaluar(entradas, ratings, unidades_):
        capturado.setdefault("primera", list(ratings))  # la primera llamada es la de por prueba
        return original(entradas, ratings, unidades_)

    motor._evaluar = evaluar
    p = Parametros()
    estados, grupos_cache = {}, []
    finales = defaultdict(dict)  # (palista, clase) -> {año: [elo, carreras]}
    for regata in regatas:
        grupos, _ = grupos_de_regata(regata)
        for grupo in grupos:
            capturado.clear()
            r = motor.procesar_grupo(grupo, estados, p)
            if not r.procesado:
                continue
            ev = grupo.entradas_evaluacion
            grupos_cache.append({
                "fecha": grupo.fecha, "clase": grupo.clase,
                "botes": [(e.palistas, e.tiempo) for e in ev],
                "unidades": unidades(grupo),
                "pruebas_unidad": [ev[u[0]].prueba_id for u in unidades(grupo)],
                "regata": regata.id,
                "elo": capturado["primera"],
            })
            for palista in r.cambios:
                fila = finales[(palista, grupo.clase)].setdefault(grupo.fecha.year, [0.0, 0])
                fila[0] = estados[palista].ratings[grupo.clase].elo
                fila[1] += 1
    motor._evaluar = original
    pickle.dump({"grupos": grupos_cache, "finales": dict(finales)}, open(CACHE, "wb"))


# ---------------------------------------------------------------------------
# Pasada 2: el modelo de tiempos
# ---------------------------------------------------------------------------

def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


ASENTADO_ANCLA, SD_ANCLA = 5, 0.03  # carreras y desviación máxima del nivel para entrar en μ diagnóstico


PASOS_HASTA_SENIOR = {"prebenjamin": 7, "benjamin": 6, "alevin": 5, "infantil": 4, "cadete": 3, "junior": 2,
                      "sub23": 1, "senior": 0, "veterano": 0}
PASOS_MAX = 7


class ModeloTiempos:
    def __init__(self, sigma, q_anual, p0_sd, transfer_sd, huber, tau=None, cats=None, mejora=0.0, arranque=0.0,
                 causal=False, multi=False, correlacion=None, ruido_correlacionado=True, correlacion_ruido=None):
        """q_anual: desviación típica del cambio de nivel en un año sin datos, por grupo de
        categoría ({"menores": .., "junior_sub23": .., "senior": .., "veterano": ..}) o un
        número para todos. tau: desviación de la dificultad de cada prueba respecto a la de
        su grupo (None: cada prueba por su cuenta).
        mejora: mejora esperada al año (log-velocidad) de un prebenjamín; la de cada categoría es
        mejora × pasos hasta senior / 7 (senior y veterano, 0). arranque: un palista nuevo empieza en
        −arranque × pasos hasta senior, en vez de en 0.
        causal: la categoría (para q, mejora y arranque) es la de la prueba que se está corriendo
        (si no tiene, la última conocida del palista; si nunca, senior), no la más corrida del año;
        no mira el futuro y no cambia al llegar regatas nuevas (la de KayakKorner)."""
        self.sigma2 = {c: v * v for c, v in sigma.items()}
        q = q_anual if isinstance(q_anual, dict) else dict.fromkeys(GRUPOS_CATEGORIA.values(), q_anual)
        self.q = {g: v ** 2 / 365.0 for g, v in q.items()}
        self.q_defecto = statistics.median(self.q.values())
        self.p0 = p0_sd ** 2
        self.pt = transfer_sd ** 2
        self.huber = huber
        self.tau2 = None if tau is None else tau * tau
        self.cats = cats or {}
        self.est = {}  # (palista, clase) -> [θ, P, última fecha, carreras]
        self.mejora, self.arranque = mejora, arranque
        self.cats_finas = categorias(finas=True) if (mejora or arranque) else {}
        self.cat_prueba = categoria_pruebas() if causal else None
        self.ultima_cat = {}  # causal: la última categoría conocida de cada palista
        # Diagnóstico para las pruebas de anclaje: por prueba, la dificultad estimada solo
        # con los botes asentados y sin compartir nada con el grupo.
        self.mus = []  # (prueba_id, μ, botes asentados)
        # Con registrar (graficas.py): por prueba, la μ con la que se actualizó, el nivel medio de los
        # botes antes de correr y cuántos eran; por palista, cada prueba y su residuo en log-tiempo
        # (> 0: más lento de lo esperado).
        self.registrar = False
        self.pruebas = {}
        self.actuaciones = defaultdict(list)

    def _preparar(self, palista, clase, fecha, cat=None):
        s = self.est.get((palista, clase))
        if s is None:
            otras = [self.est[(palista, c)] for c in CLASES if (palista, c) in self.est]
            if otras:
                mejor = min(otras, key=lambda o: o[1])
                s = [mejor[0], min(self.p0, mejor[1] + self.pt), fecha, 0]
            else:
                s = [-self.arranque * self._pasos(palista, fecha, cat), self.p0, fecha, 0]
            self.est[(palista, clase)] = s
        else:
            dias = (fecha - s[2]).days
            grupo = GRUPOS_CATEGORIA.get(cat) if cat else self.cats.get((palista, fecha.year))
            q = self.q.get(grupo, self.q_defecto)
            s[1] = min(self.p0, s[1] + q * dias)
            if self.mejora:
                s[0] += self.mejora * self._pasos(palista, fecha, cat) / PASOS_MAX * dias / 365.0
            s[2] = fecha
        return s

    def _pasos(self, palista, fecha, cat=None):
        return PASOS_HASTA_SENIOR.get(cat or self.cats_finas.get((palista, fecha.year)), 0)

    def _media_robusta(self, indices, x, w, previa=None):
        """Media ponderada con dos vueltas de Huber; `previa` = (media, varianza) a priori."""
        def media(pesos):
            num = sum(pesos[b] * x[b] for b in indices)
            den = sum(pesos.values())
            if previa:
                num += previa[0] / previa[1]
                den += 1.0 / previa[1]
            return num / den
        mu = media(w)
        for _ in range(2):
            wh = {}
            for b in indices:
                z = abs(x[b] - mu) * math.sqrt(w[b])
                wh[b] = w[b] * (1.0 if z <= self.huber else self.huber / z)
            mu = media(wh)
        return mu

    def predecir_y_actualizar(self, g):
        clase, fecha, sigma2 = g["clase"], g["fecha"], self.sigma2[g["clase"]]
        cat_bote = [None] * len(g["botes"])
        if self.cat_prueba is not None:
            for prueba_id, indices in zip(g.get("pruebas_unidad", ()), g["unidades"]):
                for b in indices:
                    cat_bote[b] = self.cat_prueba.get(prueba_id)
        def cat_de(pal, cat):
            # Causal: la de su prueba; si la prueba no tiene, la última conocida; si nunca, senior.
            if self.cat_prueba is None:
                return None
            if cat:
                self.ultima_cat[pal] = cat
            return self.ultima_cat.get(pal, "senior")

        self.previas = [[(self.est[(pal, clase)][3] if (pal, clase) in self.est else 0,
                          sum(self.est[(pal, c)][3] for c in CLASES if c != clase and (pal, c) in self.est))
                         for pal in tripulacion] for tripulacion, _ in g["botes"]]
        estados = [[self._preparar(pal, clase, fecha, cat_de(pal, cat_bote[b])) for pal in tripulacion]
                   for b, (tripulacion, _) in enumerate(g["botes"])]
        theta = [sum(s[0] for s in ss) / len(ss) for ss in estados]
        var = [sum(s[1] for s in ss) / len(ss) ** 2 for ss in estados]
        y = [math.log(t) for _, t in g["botes"]]
        x = [y[b] + theta[b] for b in range(len(y))]  # lo que cada bote dice de la dificultad
        w = [1.0 / (var[b] + sigma2) for b in range(len(y))]
        probs = {}
        for indices in g["unidades"]:
            for pos, i in enumerate(indices):
                for j in indices[pos + 1:]:
                    probs[(i, j)] = (theta[i] - theta[j]) / math.sqrt(var[i] + var[j] + 2 * sigma2)
        for prueba_id, indices in zip(g.get("pruebas_unidad", ()), g["unidades"]):
            asentados = [b for b in indices
                         if all(st[3] >= ASENTADO_ANCLA and st[1] <= SD_ANCLA ** 2 for st in estados[b])]
            if len(asentados) >= 3:
                self.mus.append((prueba_id, self._media_robusta(asentados, x, {b: w[b] for b in asentados}),
                                 len(asentados)))
        todos = [b for indices in g["unidades"] for b in indices]
        mu_grupo = None
        if self.tau2 is not None and len(g["unidades"]) > 1:
            mu_grupo = self._media_robusta(todos, x, {b: w[b] for b in todos})
        pruebas_unidad = g.get("pruebas_unidad") or [None] * len(g["unidades"])
        for prueba_id, indices in zip(pruebas_unidad, g["unidades"]):
            if len(indices) < 2 and mu_grupo is None:
                continue
            previa = (mu_grupo, self.tau2) if mu_grupo is not None else None
            mu = self._media_robusta(indices, x, {b: w[b] for b in indices}, previa)
            if self.registrar:
                self.pruebas[prueba_id] = (mu, sum(theta[b] for b in indices) / len(indices), len(indices))
            for b in indices:
                S = var[b] + sigma2
                e = y[b] - mu + theta[b]
                if self.registrar:
                    for pal in g["botes"][b][0]:
                        self.actuaciones[pal].append((prueba_id, e))
                limite = self.huber * math.sqrt(S)
                e = max(-limite, min(limite, e))
                n = len(estados[b])
                for st in estados[b]:
                    cov = st[1] / n
                    st[0] -= cov / S * e
                    st[1] = max(1e-6, st[1] - cov * cov / S)
                    st[3] += 1
        return probs, estados


# Correlación entre los niveles de las cuatro distancias (super sprint, sprint, fondo, maratón):
# la de los niveles finales de grupo_tau2_qcat_mejora_causal, palistas con ≥ 5 carreras en las dos.
CORRELACION = ((1.0, 0.821, 0.893, 0.800),
               (0.821, 1.0, 0.827, 0.781),
               (0.893, 0.827, 1.0, 0.900),
               (0.800, 0.781, 0.900, 1.0))


# La misma estimación con los datos hasta 2023 (para evaluar 2024–2026 sin haberlos visto).
CORRELACION_2023 = ((1.0, 0.833, 0.906, 0.83), (0.833, 1.0, 0.792, 0.769), (0.906, 0.792, 1.0, 0.898), (0.83, 0.769, 0.898, 1.0))


# Dentro de la cohorte (categoría y sexo del último año): los mismos niveles menos la mediana de
# su cohorte en cada distancia. Es la que importa para el radar, que compara con la categoría;
# la de todos junta las diferencias entre categorías (un infantil es más lento en todo).
CORRELACION_COHORTE = ((1.0, 0.693, 0.795, 0.66),
                       (0.693, 1.0, 0.735, 0.687),
                       (0.795, 0.735, 1.0, 0.86),
                       (0.66, 0.687, 0.86, 1.0))


def correlacion_atenuada(factor):
    """La matriz con las correlaciones multiplicadas por `factor` (0: distancias independientes)."""
    return tuple(tuple(1.0 if i == j else factor * CORRELACION[i][j] for j in range(4)) for i in range(4))


class ModeloTiemposMulti(ModeloTiempos):
    """Como ModeloTiempos, pero los cuatro niveles de cada palista son un vector con una
    covarianza 4×4: el arranque (p0·R) y lo que crece entre regatas (q·días·R) están
    correlacionados con R, así que correr una distancia mueve también las demás según su
    covarianza con ella (y sin transferencia aparte al estrenar una distancia).

    est[palista] = {"theta": [4], "P": [[4×4]], "fecha", "carreras": [4]}. Lo que devuelve
    predecir_y_actualizar por palista es la vista de la distancia corrida, [θ, P, fecha,
    carreras], como el univariante."""

    def __init__(self, *args, correlacion=CORRELACION, ruido_correlacionado=True, correlacion_ruido=None, **kwargs):
        kwargs.pop("multi", None)
        super().__init__(*args, **kwargs)
        self.R = [list(fila) for fila in correlacion]
        # Sin ruido correlacionado, la correlación solo está en el arranque: transfiere al estrenar
        # una distancia, pero lo que cambia entre regatas es independiente en cada una.
        self.Rq = self.R if ruido_correlacionado else [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        if correlacion_ruido is not None:  # otra correlación para lo que cambia entre regatas
            self.Rq = [list(fila) for fila in correlacion_ruido]
        self.indice = {c: i for i, c in enumerate(CLASES)}

    def _preparar_multi(self, palista, fecha, cat):
        st = self.est.get(palista)
        pasos = self._pasos(palista, fecha, cat)
        if st is None:
            st = {"theta": [-self.arranque * pasos] * 4,
                  "P": [[self.p0 * r for r in fila] for fila in self.R], "fecha": fecha, "carreras": [0] * 4}
            self.est[palista] = st
            return st
        dias = (fecha - st["fecha"]).days
        if dias:
            grupo = GRUPOS_CATEGORIA.get(cat) if cat else self.cats.get((palista, fecha.year))
            q = self.q.get(grupo, self.q_defecto) * dias
            P = st["P"]
            for i in range(4):
                for j in range(4):
                    P[i][j] += q * self.Rq[i][j]
            # Tope p0 en la diagonal, escalando filas y columnas (sigue siendo definida positiva).
            d = [min(1.0, math.sqrt(self.p0 / P[i][i])) for i in range(4)]
            if any(v < 1.0 for v in d):
                for i in range(4):
                    for j in range(4):
                        P[i][j] *= d[i] * d[j]
            if self.mejora:
                st["theta"] = [t + self.mejora * pasos / PASOS_MAX * dias / 365.0 for t in st["theta"]]
            st["fecha"] = fecha
        return st

    def predecir_y_actualizar(self, g):
        clase, fecha, sigma2 = g["clase"], g["fecha"], self.sigma2[g["clase"]]
        k = self.indice[clase]
        cat_bote = [None] * len(g["botes"])
        if self.cat_prueba is not None:
            for prueba_id, indices in zip(g.get("pruebas_unidad", ()), g["unidades"]):
                for b in indices:
                    cat_bote[b] = self.cat_prueba.get(prueba_id)

        def cat_de(pal, cat):
            if self.cat_prueba is None:
                return None
            if cat:
                self.ultima_cat[pal] = cat
            return self.ultima_cat.get(pal, "senior")

        self.previas = [[(self.est[pal]["carreras"][k] if pal in self.est else 0,
                          sum(self.est[pal]["carreras"]) - self.est[pal]["carreras"][k] if pal in self.est else 0)
                         for pal in tripulacion] for tripulacion, _ in g["botes"]]
        sts = [[self._preparar_multi(pal, fecha, cat_de(pal, cat_bote[b])) for pal in tripulacion]
               for b, (tripulacion, _) in enumerate(g["botes"])]
        theta = [sum(st["theta"][k] for st in ss) / len(ss) for ss in sts]
        var = [sum(st["P"][k][k] for st in ss) / len(ss) ** 2 for ss in sts]
        y = [math.log(t) for _, t in g["botes"]]
        x = [y[b] + theta[b] for b in range(len(y))]
        w = [1.0 / (var[b] + sigma2) for b in range(len(y))]
        probs = {}
        for indices in g["unidades"]:
            for pos, i in enumerate(indices):
                for j in indices[pos + 1:]:
                    probs[(i, j)] = (theta[i] - theta[j]) / math.sqrt(var[i] + var[j] + 2 * sigma2)
        for prueba_id, indices in zip(g.get("pruebas_unidad", ()), g["unidades"]):
            asentados = [b for b in indices
                         if all(st["carreras"][k] >= ASENTADO_ANCLA and st["P"][k][k] <= SD_ANCLA ** 2 for st in sts[b])]
            if len(asentados) >= 3:
                self.mus.append((prueba_id, self._media_robusta(asentados, x, {b: w[b] for b in asentados}),
                                 len(asentados)))
        todos = [b for indices in g["unidades"] for b in indices]
        mu_grupo = None
        if self.tau2 is not None and len(g["unidades"]) > 1:
            mu_grupo = self._media_robusta(todos, x, {b: w[b] for b in todos})
        pruebas_unidad = g.get("pruebas_unidad") or [None] * len(g["unidades"])
        for prueba_id, indices in zip(pruebas_unidad, g["unidades"]):
            if len(indices) < 2 and mu_grupo is None:
                continue
            previa = (mu_grupo, self.tau2) if mu_grupo is not None else None
            mu = self._media_robusta(indices, x, {b: w[b] for b in indices}, previa)
            if self.registrar:
                self.pruebas[prueba_id] = (mu, sum(theta[b] for b in indices) / len(indices), len(indices))
            for b in indices:
                S = var[b] + sigma2
                e = y[b] - mu + theta[b]
                if self.registrar:
                    for pal in g["botes"][b][0]:
                        self.actuaciones[pal].append((prueba_id, e))
                limite = self.huber * math.sqrt(S)
                e = max(-limite, min(limite, e))
                n = len(sts[b])
                for st in sts[b]:
                    P = st["P"]
                    col = [P[i][k] / n for i in range(4)]
                    st["theta"] = [st["theta"][i] - col[i] / S * e for i in range(4)]
                    for i in range(4):
                        for j in range(4):
                            P[i][j] -= col[i] * col[j] / S
                    P[k][k] = max(1e-6, P[k][k])
                    st["carreras"][k] += 1
        estados = [[[st["theta"][k], st["P"][k][k], st["fecha"], st["carreras"][k]] for st in ss] for ss in sts]
        return probs, estados


def nuevo_modelo(config, cats):
    if config.get("multi"):
        return ModeloTiemposMulti(**config, cats=cats)
    return ModeloTiempos(**config, cats=cats)


GRUPOS_CATEGORIA = {"prebenjamin": "menores", "benjamin": "menores", "alevin": "menores", "infantil": "menores",
                    "cadete": "menores", "junior": "junior_sub23", "sub23": "junior_sub23", "senior": "senior",
                    "veterano": "veterano"}


_CATEGORIAS = {}


def categorias(finas=False):
    """{(palista, año): grupo de su categoría más corrida ese año} (con finas, la categoría)."""
    if finas in _CATEGORIAS:
        return _CATEGORIAS[finas]
    cuenta = defaultdict(lambda: defaultdict(int))
    for r in json.load(open(DATOS)):
        año = int(r["fecha"][:4])
        for p in r["pruebas"]:
            g = p.get("categoria") if finas and p.get("categoria") in GRUPOS_CATEGORIA else \
                GRUPOS_CATEGORIA.get(p.get("categoria"))
            if g:
                for res in p["results"]:
                    for x in res["lista_palista_id"]:
                        if x:
                            cuenta[(int(x), año)][g] += 1
    _CATEGORIAS[finas] = {k: max(v, key=v.get) for k, v in cuenta.items()}
    return _CATEGORIAS[finas]


def categoria_pruebas():
    """{prueba: categoría}."""
    if "pruebas" not in _CATEGORIAS:
        _CATEGORIAS["pruebas"] = {int(p["prueba_id"]): p.get("categoria") for r in json.load(open(DATOS))
                                  for p in r["pruebas"] if p.get("categoria") in GRUPOS_CATEGORIA}
    return _CATEGORIAS["pruebas"]


def deriva(finales, cats):
    """{grupo de categoría: (media del cambio entre temporadas seguidas, n)} desde la 3.ª temporada.
    La categoría es la del segundo año; se cuenta también «veterano_estable» (veterano los dos años)."""
    cambios = defaultdict(list)
    for (palista, clase), años in finales.items():
        primero = min(años)
        validos = {a: v for a, v in años.items() if v[1] >= 3}
        for a in sorted(validos):
            if a + 1 in validos and a + 1 - primero >= 2:
                g = cats.get((palista, a + 1))
                if g:
                    delta = validos[a + 1][0] - validos[a][0]
                    cambios[g].append(delta)
                    if g == "veterano" and cats.get((palista, a)) == "veterano":
                        cambios["veterano_estable"].append(delta)
    return {g: (statistics.mean(v), len(v)) for g, v in sorted(cambios.items())}


def brecha_distancias(finales, minimo=5):
    """Lo que el radar distingue entre distancias. Para los palistas con ≥ `minimo` carreras en
    al menos dos distancias: su nivel final en cada una en dispersiones de la población de esa
    distancia ((θ − mediana) / ((p90 − p10) / 2,563), con los que tienen ≥ `minimo` carreras), y
    la diferencia entre la mejor y la peor. También la correlación entre distancias."""
    niveles = defaultdict(dict)
    for (palista, clase), años in finales.items():
        if sum(n for _, n in años.values()) >= minimo:
            niveles[palista][clase] = años[max(años)][0]
    escala = {}
    for c in CLASES:
        v = sorted(n[c] for n in niveles.values() if c in n)
        q = lambda p: v[int(p * (len(v) - 1))]
        escala[c] = (statistics.median(v), (q(0.9) - q(0.1)) / 2.563)
    brechas = []
    for n in niveles.values():
        if len(n) >= 2:
            z = [(n[c] - escala[c][0]) / escala[c][1] for c in n]
            brechas.append(max(z) - min(z))
    correl = {}
    for i, a in enumerate(CLASES):
        for b in CLASES[i + 1:]:
            pares = [(n[a], n[b]) for n in niveles.values() if a in n and b in n]
            xs, ys = zip(*pares)
            ma, mb = statistics.mean(xs), statistics.mean(ys)
            correl[f"{a}-{b}"] = sum((x - ma) * (y - mb) for x, y in pares) / math.sqrt(
                sum((x - ma) ** 2 for x in xs) * sum((y - mb) ** 2 for y in ys))
    return {"palistas": len(brechas), "brecha_mediana": statistics.median(brechas),
            "brecha_media": statistics.mean(brechas),
            "brecha_p90": sorted(brechas)[int(0.9 * (len(brechas) - 1))], "correlacion": correl}


def dispersion(finales):
    por_año = defaultdict(list)
    for (palista, clase), años in finales.items():
        primero = min(años)
        for a, (valor, n) in años.items():
            if n >= 3 and a - primero >= 2 and clase == "fondo":
                por_año[a].append(valor)
    return {a: statistics.pstdev(v) for a, v in sorted(por_año.items()) if len(v) >= 50}


def log_loss(pares, prob, s):
    tot = 0.0
    for d, real in pares:
        p = min(1 - 1e-12, max(1e-12, prob(s * d)))
        tot -= real * math.log(p) + (1 - real) * math.log(1 - p)
    return tot / len(pares)


def ajustar_s(pares, prob):
    """s que minimiza el log loss de prob(s·d) (sección áurea en [0.05, 5])."""
    a, b = 0.05, 5.0
    r = (math.sqrt(5) - 1) / 2
    c, d = b - r * (b - a), a + r * (b - a)
    fc, fd = log_loss(pares, prob, c), log_loss(pares, prob, d)
    for _ in range(40):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - r * (b - a); fc = log_loss(pares, prob, c)
        else:
            a, c, fc = c, d, fd
            d = a + r * (b - a); fd = log_loss(pares, prob, d)
    return (a + b) / 2


def logistica(d):
    return 1 / (1 + 10 ** (-d / 400))


class Metr:
    def __init__(self):
        self.n = self.dec = self.ok = 0
        self.ll = self.brier = 0.0

    def add(self, p, real):
        self.n += 1
        if real != 0.5:
            self.dec += 1
            self.ok += (p > 0.5) == (real == 1.0)
        self.brier += (p - real) ** 2
        p = min(1 - 1e-15, max(1e-15, p))
        self.ll -= real * math.log(p) + (1 - real) * math.log(1 - p)

    def resumen(self):
        return {"pares": self.n, "acierto": self.ok / self.dec, "log_loss": self.ll / self.n, "brier": self.brier / self.n}


CORTE = 2019  # temperatura por distancia ajustada hasta este año, evaluada después


def correr(cache, cats, config, completo=True):
    """Métricas en bruto sobre todos los pares; calibradas (una escala s por distancia y
    sistema, ajustada con los pares hasta CORTE) sobre una muestra del 10 % de los pares
    decisivos, evaluadas antes (ajuste) y después (prueba) del corte."""
    modelo = nuevo_modelo(config, cats)
    random.seed(0)
    m_elo, m_t = Metr(), Metr()
    muestra = []  # (año, clase, d_elo, z_tiempos, real)
    finales = defaultdict(dict)
    for g in cache["grupos"]:
        zs, estados = modelo.predecir_y_actualizar(g)
        elo = g["elo"]
        año = g["fecha"].year
        for (i, j), z in zs.items():
            ti, tj = g["botes"][i][1], g["botes"][j][1]
            real = 1.0 if ti < tj else 0.0 if ti > tj else 0.5
            m_elo.add(esperado(elo[i], elo[j]), real); m_t.add(phi(z), real)
            if real != 0.5 and random.random() < 0.1:
                # Lo que llevaba cada integrante de los dos botes antes de correr: carreras en esta
                # distancia (la mínima) y si alguno la estrena habiendo corrido ya otras (≥ 3).
                previas = modelo.previas[i] + modelo.previas[j]
                minimo = min(n for n, _ in previas)
                estrena = any(n == 0 and otras >= 3 for n, otras in previas)
                muestra.append((año, g["clase"], elo[i] - elo[j], z, real, minimo, estrena))
        if completo:
            for (tripulacion, _), ss in zip(g["botes"], estados):
                for pal, s in zip(tripulacion, ss):
                    fila = finales[(pal, g["clase"])].setdefault(año, [0.0, 0])
                    fila[0] = s[0]; fila[1] += 1

    sistemas = {"elo": (2, logistica), "tiempos": (3, phi)}
    escalas = {nombre: {c: ajustar_s([(x[k], x[4]) for x in muestra if x[1] == c and x[0] <= CORTE], prob)
                        for c in CLASES} for nombre, (k, prob) in sistemas.items()}

    def calibradas(nombre, filtro):
        k, prob = sistemas[nombre]
        m = Metr()
        for x in muestra:
            if filtro(x):
                m.add(prob(escalas[nombre][x[1]] * x[k]), x[4])
        return m.resumen()

    res = {"config": config, "bruto": {"elo": m_elo.resumen(), "tiempos": m_t.resumen()},
           "escala_por_clase": escalas,
           "calibrado_ajuste": {n: calibradas(n, lambda x: x[0] <= CORTE) for n in sistemas},
           "calibrado_prueba": {n: calibradas(n, lambda x: x[0] > CORTE) for n in sistemas},
           # Pares con pocas carreras en la distancia (2020–2026): alguno con menos de 3, alguno
           # que la estrena habiendo corrido otras, y los dos botes con 3 o más.
           "calibrado_prueba_pocas": {n: calibradas(n, lambda x: x[0] > CORTE and x[5] < 3) for n in sistemas},
           "calibrado_prueba_estrena": {n: calibradas(n, lambda x: x[0] > CORTE and x[6]) for n in sistemas},
           "calibrado_prueba_asentados": {n: calibradas(n, lambda x: x[0] > CORTE and x[5] >= 3) for n in sistemas},
           # 2024–2026: lo que ninguna correlación estimada hasta 2023 ha visto.
           "calibrado_2024": {n: calibradas(n, lambda x: x[0] >= 2024) for n in sistemas},
           "calibrado_2024_pocas": {n: calibradas(n, lambda x: x[0] >= 2024 and x[5] < 3) for n in sistemas},
           "calibrado_2024_estrena": {n: calibradas(n, lambda x: x[0] >= 2024 and x[6]) for n in sistemas}}
    if completo:
        res["calibrado_prueba_por_clase"] = {
            c: {n: calibradas(n, lambda x, c=c: x[0] > CORTE and x[1] == c) for n in sistemas} for c in CLASES}
        # s por año sobre las probabilidades ya calibradas por distancia: si se aleja de 1
        # en algunos años, la misma diferencia no significa lo mismo esos años.
        años = sorted({x[0] for x in muestra})
        res["escala_por_año"] = {
            a: {n: ajustar_s([(escalas[n][x[1]] * x[k], x[4]) for x in muestra if x[0] == a], prob)
                for n, (k, prob) in sistemas.items()} for a in años}
        res["deriva_elo"] = deriva(cache["finales"], cats)
        res["deriva_tiempos_pct"] = {g: (100 * m, n) for g, (m, n) in deriva(finales, cats).items()}
        res["dispersion_fondo_elo"] = dispersion(cache["finales"])
        res["dispersion_fondo_tiempos_pct"] = {a: 100 * v for a, v in dispersion(finales).items()}
        res["brecha_distancias"] = brecha_distancias(finales)
    return res


BASE = dict(sigma={"super_sprint": 0.02, "sprint": 0.03, "fondo": 0.035, "maraton": 0.04},
            q_anual=0.02, p0_sd=0.08, transfer_sd=0.03, huber=2.5)

Q_CATEGORIA = {"menores": 0.06, "junior_sub23": 0.04, "senior": 0.02, "veterano": 0.02}
VARIANTES = {
    "por_prueba": BASE,
    "grupo_tau1": dict(BASE, tau=0.01),
    "grupo_tau1_qcat": dict(BASE, tau=0.01, q_anual=Q_CATEGORIA),
    "grupo_tau05_qcat": dict(BASE, tau=0.005, q_anual=Q_CATEGORIA),
    "grupo_tau2_qcat": dict(BASE, tau=0.02, q_anual=Q_CATEGORIA),
    # Mejora esperada y arranque de los menores calibrados con las anclas (calibrar.py).
    "grupo_tau2_qcat_mejora": dict(BASE, tau=0.02, q_anual=Q_CATEGORIA, mejora=0.11, arranque=0.07),
    # Lo mismo con la categoría de la prueba que se corre (sin mirar el futuro): la de KayakKorner.
    "grupo_tau2_qcat_mejora_causal": dict(BASE, tau=0.02, q_anual=Q_CATEGORIA, mejora=0.11, arranque=0.07,
                                          causal=True),
    # Transferencia entre distancias: niveles correlacionados (ModeloTiemposMulti).
    "multi_R": dict(BASE, tau=0.02, q_anual=Q_CATEGORIA, mejora=0.11, arranque=0.07, causal=True, multi=True,
                    correlacion=CORRELACION),
    "multi_R_prior": dict(BASE, tau=0.02, q_anual=Q_CATEGORIA, mejora=0.11, arranque=0.07, causal=True,
                          multi=True, correlacion=CORRELACION, ruido_correlacionado=False),
    "multi_Rcoh": dict(BASE, tau=0.02, q_anual=Q_CATEGORIA, mejora=0.11, arranque=0.07, causal=True,
                       multi=True, correlacion=CORRELACION_COHORTE),
    "multi_R_ruido_mitad": dict(BASE, tau=0.02, q_anual=Q_CATEGORIA, mejora=0.11, arranque=0.07, causal=True,
                                multi=True, correlacion=CORRELACION, correlacion_ruido=correlacion_atenuada(0.5)),
    "multi_R2023": dict(BASE, tau=0.02, q_anual=Q_CATEGORIA, mejora=0.11, arranque=0.07, causal=True,
                        multi=True, correlacion=CORRELACION_2023),
    "multi_R_mitad": dict(BASE, tau=0.02, q_anual=Q_CATEGORIA, mejora=0.11, arranque=0.07, causal=True,
                          multi=True, correlacion=correlacion_atenuada(0.5)),
    "multi_independientes": dict(BASE, tau=0.02, q_anual=Q_CATEGORIA, mejora=0.11, arranque=0.07, causal=True,
                                 multi=True, correlacion=correlacion_atenuada(0.0)),
}

if __name__ == "__main__":
    if not CACHE.exists():
        pasada_elo()
    cache = pickle.load(open(CACHE, "rb"))
    cats = categorias()
    nombres = sys.argv[1:] or list(VARIANTES)
    for nombre in nombres:
        res = correr(cache, cats, VARIANTES[nombre])
        json.dump(res, open(AQUI / f"results/{nombre}.json", "w"), indent=1, default=str)
        print(nombre, "listo", flush=True)
