"""Starting-rating strategies for new athletes (KayakKorner engine variant).

Estrategias de arranque de un palista nuevo (su primera regata).

base:        1500
rend:        1500 + f·(rendimiento − 1500)              (f = FRACCION; 1 = solo rendimiento)
prueba:      media de los participantes conocidos de su prueba (si no hay, del grupo; si no, 1500)
cohorte:     media de los asentados de su categoría y sexo activos el último año
cohorte_rend: cohorte + f·(rendimiento − cohorte)

El rendimiento es el rating con el que el resultado del bote habría sido el
esperado frente a sus rivales (bisección entre 1100 y 2100; los nuevos del
mismo grupo se estiman a la vez). La predicción de esa primera regata se hace
antes, con 1500, como siempre.
"""
from collections import defaultdict
from datetime import timedelta

from prediction.elo.motor import (CLASES, EstadoPalista, Metricas, ResultadoGrupo, _aplicar, _cambio_palista,
                                  _comparten, _evaluar, _media, esperado, preparar, resultado_par)

MIN, MAX = 1100.0, 2100.0
ESTRATEGIA = "base"
FRACCION = 0.3
META = {}  # prueba_id -> (categoria, sexo)

# Cohortes: (categoria, sexo) -> {palista: fecha de su última carrera en ella}
_miembros = defaultdict(dict)
_cohorte_de = {}


# Pares del grupo fusionado según lo que comparan: {tipo: [pares, aciertos, decisivos, log_loss, sesgo]}
CRUZADOS = {}
PRIMERA = {}  # palista -> fecha de su primera regata
_ORDEN = ("prebenjamin", "benjamin", "alevin", "infantil", "cadete", "junior", "sub23", "senior", "veterano")


def reiniciar():
    _miembros.clear()
    _cohorte_de.clear()
    CRUZADOS.clear()
    PRIMERA.clear()


def _sumar(tipo, prob, real):
    """prob: probabilidad de que gane el bote «a»; real: 1, 0,5 o 0 para ese mismo bote."""
    import math
    fila = CRUZADOS.setdefault(tipo, [0, 0, 0, 0.0, 0.0, 0.0])
    q = min(1 - 1e-15, max(1e-15, prob))
    fila[0] += 1
    if real != 0.5:
        fila[2] += 1
        fila[1] += (prob > 0.5) == (real == 1.0)
    fila[3] += -(real * math.log(q) + (1 - real) * math.log(1 - q))
    fila[4] += prob
    fila[5] += real


def _novato(entrada, fecha):
    return any((fecha - PRIMERA[x]).days < 365 for x in entrada.palistas if x in PRIMERA)


def _medir_cruzados(entradas, ratings, fecha):
    for i in range(len(entradas)):
        for j in range(i + 1, len(entradas)):
            a, b = entradas[i], entradas[j]
            if _comparten(a, b):
                continue
            prob = esperado(ratings[i], ratings[j])
            real = resultado_par(a, b)
            ma, mb = META.get(a.prueba_id, (None, None)), META.get(b.prueba_id, (None, None))
            # Sexo distinto, orientado a la mujer: P(gana la mujer) frente a lo que pasa.
            if {ma[1], mb[1]} == {"femenino", "masculino"}:
                if ma[1] == "femenino":
                    _sumar("sexo distinto", prob, real)
                else:
                    _sumar("sexo distinto", 1 - prob, 1 - real)
            # Categoría distinta, orientado a la más joven.
            if ma[0] in _ORDEN and mb[0] in _ORDEN and ma[0] != mb[0]:
                if _ORDEN.index(ma[0]) < _ORDEN.index(mb[0]):
                    _sumar("categoría distinta", prob, real)
                else:
                    _sumar("categoría distinta", 1 - prob, 1 - real)
            # Novatos (primer año) frente a asentados.
            tipo = "con novato" if _novato(a, fecha) or _novato(b, fecha) else "entre asentados"
            _sumar(tipo, prob, real)


def _rendimiento(i, entradas, ratings):
    real, rivales = 0.0, []
    for j, rival in enumerate(entradas):
        if i == j or _comparten(entradas[i], rival):
            continue
        real += resultado_par(entradas[i], rival)
        rivales.append(ratings[j])
    if not rivales:
        return None
    objetivo = real / len(rivales)
    bajo, alto = MIN, MAX
    for _ in range(40):
        medio = (bajo + alto) / 2
        if sum(esperado(medio, r) for r in rivales) / len(rivales) < objetivo:
            bajo = medio
        else:
            alto = medio
    return (bajo + alto) / 2


def _media_cohorte(cohorte, clase, fecha, estados):
    desde = fecha - timedelta(days=365)
    valores = [estados[p].ratings[clase].elo for p, ultima in _miembros[cohorte].items()
               if ultima >= desde and estados[p].ratings[clase].n_efectiva >= 5]
    return sum(valores) / len(valores) if len(valores) >= 5 else None


def _rendimientos_por_palista(grupo, nuevos, congelados, estados):
    """{palista nuevo: rendimiento individual} (el del barco, descontando a los compañeros conocidos)."""
    con_nuevos = [i for i, e in enumerate(grupo.entradas) if set(e.palistas) & nuevos]
    if not con_nuevos or len(con_nuevos) == len(grupo.entradas):
        return {}
    estimados = list(congelados)
    for _ in range(10):
        for i in con_nuevos:
            r = _rendimiento(i, grupo.entradas, estimados)
            if r is not None:
                estimados[i] = r
    salida = {}
    for i in con_nuevos:
        entrada = grupo.entradas[i]
        conocidos = [estados[x].efectivo(grupo.clase, grupo.tipo, grupo.barco) for x in entrada.palistas if x not in nuevos]
        n_nuevos = len(entrada.palistas) - len(conocidos)
        valor = max(MIN, min(MAX, (estimados[i] * len(entrada.palistas) - sum(conocidos)) / n_nuevos))
        for x in entrada.palistas:
            if x in nuevos:
                salida[x] = valor
    return salida


def _arranques(grupo, nuevos, congelados, estados):
    if ESTRATEGIA == "base" or not nuevos:
        return {}
    rend = _rendimientos_por_palista(grupo, nuevos, congelados, estados) if "rend" in ESTRATEGIA else {}
    salida = {}
    if ESTRATEGIA == "rend":
        return {x: 1500.0 + FRACCION * (r - 1500.0) for x, r in rend.items()}
    if ESTRATEGIA == "prueba":
        conocidas = [(e, congelados[i]) for i, e in enumerate(grupo.entradas) if not set(e.palistas) & nuevos]
        media_grupo = sum(r for _, r in conocidas) / len(conocidas) if conocidas else None
        for e in grupo.entradas:
            de_su_prueba = [r for c, r in conocidas if c.prueba_id == e.prueba_id]
            valor = sum(de_su_prueba) / len(de_su_prueba) if de_su_prueba else media_grupo
            if valor is not None:
                for x in e.palistas:
                    if x in nuevos:
                        salida[x] = valor
        return salida
    # cohorte / cohorte_rend
    for e in grupo.entradas:
        cohorte = META.get(e.prueba_id)
        base = _media_cohorte(cohorte, grupo.clase, grupo.fecha, estados) if cohorte and None not in cohorte else None
        for x in e.palistas:
            if x not in nuevos:
                continue
            if ESTRATEGIA == "cohorte":
                if base is not None:
                    salida[x] = base
            else:
                ancla = base if base is not None else 1500.0
                if x in rend:
                    salida[x] = ancla + FRACCION * (rend[x] - ancla)
                elif base is not None:
                    salida[x] = base
    return salida


def procesar_grupo(grupo, estados, p):
    if not grupo.usable:
        return ResultadoGrupo(grupo, False, Metricas(), Metricas())
    nuevos = set()
    for entrada in grupo.entradas:
        for palista in entrada.palistas:
            estado = estados.get(palista)
            if estado is None:
                estado = estados[palista] = EstadoPalista()
                nuevos.add(palista)
            if estado.tipo.principal is None:
                estado.tipo.principal = grupo.tipo
            if estado.barco.principal is None:
                estado.barco.principal = grupo.barco
            preparar(estado, grupo.fecha, p)

    def ratings(entradas):
        return [sum(estados[x].efectivo(grupo.clase, grupo.tipo, grupo.barco) for x in e.palistas) / len(e.palistas)
                for e in entradas]

    evaluacion = grupo.entradas_evaluacion
    por_prueba = _evaluar(evaluacion, ratings(evaluacion), [
        [i for i, e in enumerate(evaluacion) if e.indice_prueba == indice]
        for indice in sorted({e.indice_prueba for e in evaluacion})])
    congelados = ratings(grupo.entradas)
    agrupado = _evaluar(grupo.entradas, congelados, [list(range(len(grupo.entradas)))])
    for entrada in grupo.entradas:
        for x in entrada.palistas:
            PRIMERA.setdefault(x, grupo.fecha)
    _medir_cruzados(grupo.entradas, congelados, grupo.fecha)

    arranques = _arranques(grupo, nuevos, congelados, estados)
    if arranques:
        for x, valor in arranques.items():
            for c in CLASES:
                estados[x].ratings[c].elo = valor
        congelados = ratings(grupo.entradas)

    n = len(grupo.entradas)
    k = p.k * max(p.campo_minimo, min(p.campo_maximo, (float(n) / p.campo_referencia) ** p.campo_alfa))
    cambios = {}
    for i, entrada in enumerate(grupo.entradas):
        real = esperados = 0.0
        rivales = 0
        for j, rival in enumerate(grupo.entradas):
            if i == j or _comparten(entrada, rival):
                continue
            real += resultado_par(entrada, rival)
            esperados += esperado(congelados[i], congelados[j])
            rivales += 1
        delta = k * (real / rivales - esperados / rivales) if rivales else 0.0
        for x in entrada.palistas:
            cambio = _cambio_palista(estados[x], delta, len(entrada.palistas), grupo.clase, grupo.tipo, grupo.barco, p)
            cambio.pruebas_ids = entrada.pruebas_ids
            cambios.setdefault(x, []).append(cambio)
    medias = {x: _media(lista) for x, lista in cambios.items()}
    for x, cambio in medias.items():
        _aplicar(estados[x], cambio, grupo, p)

    # Cohortes: la última categoría y sexo en que ha corrido cada uno.
    for entrada in grupo.entradas:
        cohorte = META.get(entrada.prueba_id)
        if not cohorte or None in cohorte:
            continue
        for x in entrada.palistas:
            anterior = _cohorte_de.get(x)
            if anterior and anterior != cohorte:
                _miembros[anterior].pop(x, None)
            _cohorte_de[x] = cohorte
            _miembros[cohorte][x] = grupo.fecha
    return ResultadoGrupo(grupo, True, por_prueba, agrupado, medias)
