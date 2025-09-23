from collections import deque, defaultdict
import heapq
import math

# La clase Nodo
class Nodo:
    def __init__(self, estado, padre=None, accion=None, costo_camino=0):
        self.estado = estado
        self.padre = padre
        self.accion = accion
        self.costo_camino = costo_camino  # g(n): costo desde el inicio

    def __lt__(self, otro):
        # Esta comparación se usa para el "heap", compara los nodos por su costo de camino
        return self.costo_camino < otro.costo_camino

# La clase Nodo con valor f() para A* y búsqueda voraz
class NodoAStar:
    def __init__(self, estado, padre=None, accion=None, costo_camino=0, heuristica=0):
        self.estado = estado
        self.padre = padre
        self.accion = accion
        self.costo_camino = costo_camino  # g(n)
        self.heuristica = heuristica      # h(n)
        self.costo_total = costo_camino + heuristica # f(n) = g(n) + h(n)

    def __lt__(self, otro):
        # El heap prioriza nodos con el menor costo total f(n)
        return self.costo_total < otro.costo_total

# Reconstruir el camino desde el nodo objetivo
def reconstruir_camino(nodo):
    camino = []
    while nodo:
        # Añade la acción y el estado al camino
        camino.append((nodo.accion, nodo.estado))
        nodo = nodo.padre
    # El camino se reconstruye desde el final, así que lo invertimos
    return list(reversed(camino))

# Algoritmo de Búsqueda A*
def busqueda_a_estrella(estado_inicial, es_objetivo, obtener_sucesores, heuristica):
    frontera = []
    # Usamos un heapq para una cola de prioridad, prioriza los nodos con menor f(n)
    heapq.heappush(frontera, NodoAStar(estado_inicial, costo_camino=0, heuristica=heuristica(estado_inicial)))
    
    # visitados almacena los estados visitados y el costo mínimo encontrado
    visitados = {estado_inicial: 0}
    nodos_expandidos = 0

    while frontera:
        nodo_actual = heapq.heappop(frontera)
        nodos_expandidos += 1
        
        if es_objetivo(nodo_actual.estado):
            return reconstruir_camino(nodo_actual), nodos_expandidos
        
        # Si ya hemos encontrado un camino más corto a este estado, lo ignoramos
        if nodo_actual.costo_camino > visitados.get(nodo_actual.estado, float('inf')):
            continue

        for accion, estado_siguiente, costo in obtener_sucesores(nodo_actual.estado):
            nuevo_costo = nodo_actual.costo_camino + costo
            
            # Solo procesamos el sucesor si es la primera vez que lo vemos o si encontramos un camino más corto
            if nuevo_costo < visitados.get(estado_siguiente, float('inf')):
                visitados[estado_siguiente] = nuevo_costo
                nuevo_nodo = NodoAStar(estado_siguiente, nodo_actual, accion, nuevo_costo, heuristica(estado_siguiente))
                heapq.heappush(frontera, nuevo_nodo)
    
    return None, nodos_expandidos

# Búsqueda Voraz (Greedy Best-First Search)
def busqueda_voraz(estado_inicial, es_objetivo, obtener_sucesores, heuristica):
    frontera = []
    # La prioridad es solo la heurística: h(n)
    heapq.heappush(frontera, (heuristica(estado_inicial), Nodo(estado_inicial)))
    
    visitados = set([estado_inicial])
    nodos_expandidos = 0

    while frontera:
        # El heapq devuelve la tupla con menor valor, por lo que la heurística es el primer elemento de la tupla
        _, nodo_actual = heapq.heappop(frontera)
        nodos_expandidos += 1
        
        if es_objetivo(nodo_actual.estado):
            return reconstruir_camino(nodo_actual), nodos_expandidos
        
        for accion, estado_siguiente, costo in obtener_sucesores(nodo_actual.estado):
            if estado_siguiente not in visitados:
                visitados.add(estado_siguiente)
                nuevo_nodo = Nodo(estado_siguiente, nodo_actual, accion, nodo_actual.costo_camino + costo)
                # La prioridad es la heurística del nuevo estado
                heapq.heappush(frontera, (heuristica(estado_siguiente), nuevo_nodo))
    
    return None, nodos_expandidos

# Funciones de heurística existentes, son compatibles con A* y Búsqueda Voraz
def heuristica_manhattan(estado_actual, estado_objetivo):
    # Asume que los estados son tuplas (x, y)
    return abs(estado_actual[0] - estado_objetivo[0]) + abs(estado_actual[1] - estado_objetivo[1])

def heuristica_euclidiana(estado_actual, estado_objetivo):
    return math.sqrt((estado_actual[0] - estado_objetivo[0])**2 + (estado_actual[1] - estado_objetivo[1])**2)