# main.py

# Importa las funciones de busqueda y la clase Nodo del archivo anterior
from Busqueda import (
    busqueda_a_estrella,
    busqueda_voraz,
    busqueda_anchura,
    heuristica_manhattan,
    reconstruir_camino
)
import random

# Definicion del problema: Un simple problema de puzzle (1D)
# Estado: una tupla de numeros que representa el tablero
# Por ejemplo, (1, 2, 3, 4, 5, 6, 7, 8, 0)
# donde 0 es el espacio vacio

def es_objetivo(estado):
    """Verifica si el estado es el estado objetivo."""
    # En este ejemplo, el estado objetivo es el mismo que el inicial ordenado
    estado_objetivo = tuple(sorted(list(estado)))
    return estado == estado_objetivo

def obtener_sucesores(estado):
    """
    Genera los posibles estados sucesores desde un estado dado.
    Para este puzzle 1D, un movimiento es intercambiar el 0 con un vecino.
    """
    sucesores = []
    lista_estado = list(estado)
    indice_vacio = lista_estado.index(0)

    # Movimiento hacia la izquierda
    if indice_vacio > 0:
        nueva_lista = lista_estado[:]
        nueva_lista[indice_vacio], nueva_lista[indice_vacio - 1] = nueva_lista[indice_vacio - 1], nueva_lista[indice_vacio]
        sucesores.append(("mover_izquierda", tuple(nueva_lista), 1)) # acción, estado, costo
    
    # Movimiento hacia la derecha
    if indice_vacio < len(lista_estado) - 1:
        nueva_lista = lista_estado[:]
        nueva_lista[indice_vacio], nueva_lista[indice_vacio + 1] = nueva_lista[indice_vacio + 1], nueva_lista[indice_vacio]
        sucesores.append(("mover_derecha", tuple(nueva_lista), 1))

    return sucesores

# Definicion de la heuristica para el problema del puzzle
def heuristica_puzzle_1d(estado_actual, estado_objetivo):
    """
    Heuristica de 'piezas fuera de lugar'.
    Cuenta cuantos numeros no estan en su posicion final.
    """
    estado_objetivo = tuple(sorted(list(estado_actual)))
    # El valor heuristico es la cantidad de elementos que no coinciden
    return sum(1 for a, b in zip(estado_actual, estado_objetivo) if a != b)

# --- Logica principal de la aplicacion ---
def run_app():
    # Define el estado inicial y objetivo para el problema
    estado_objetivo = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    
    # Crea un estado inicial aleatorio y desordenado
    estado_inicial_lista = list(estado_objetivo)
    random.shuffle(estado_inicial_lista)
    estado_inicial = tuple(estado_inicial_lista)

    print(f"Estado inicial del puzzle: {estado_inicial}")
    print(f"Estado objetivo del puzzle: {estado_objetivo}\n")

    # --- Ejecuta los algoritmos ---
    print("Iniciando busqueda con A*...")
    # Se pasa la funcion heuristica como un argumento a la funcion de busqueda
    # A* necesita una funcion de heuristica que reciba el estado actual
    def heuristica_a_star(estado):
        return heuristica_puzzle_1d(estado, estado_objetivo)
    
    solucion_a_star, nodos_a_star = busqueda_a_estrella(estado_inicial, es_objetivo, obtener_sucesores, heuristica_a_star)
    
    if solucion_a_star:
        print(f"Solucion encontrada por A* en {len(solucion_a_star) - 1} pasos.")
        print(f"Nodos expandidos por A*: {nodos_a_star}")
        # Opcional: imprimir los pasos de la solucion
        # for accion, estado in solucion_a_star:
        #     print(f"-> Accion: {accion}, Estado: {estado}")
    else:
        print(" A* no encontro una solucion.")

    print("\n-----------------------------------\n")

    print("Iniciando búsqueda Voraz...")
    # Busqueda Voraz tambien necesita la heuristica
    def heuristica_voraz(estado):
        return heuristica_puzzle_1d(estado, estado_objetivo)

    solucion_voraz, nodos_voraz = busqueda_voraz(estado_inicial, es_objetivo, obtener_sucesores, heuristica_voraz)
    
    if solucion_voraz:
        print(f" Solucion encontrada por busqueda Voraz en {len(solucion_voraz) - 1} pasos.")
        print(f"Nodos expandidos por busqueda Voraz: {nodos_voraz}")
    else:
        print(" La busqueda Voraz no encontro una solucion.")

    print("\n-----------------------------------\n")

    print("Iniciando busqueda en Anchura (para comparar)...")
    solucion_anchura, nodos_anchura = busqueda_anchura(estado_inicial, es_objetivo, obtener_sucesores)
    
    if solucion_anchura:
        print(f" Solucion encontrada por busqueda en Anchura en {len(solucion_anchura) - 1} pasos.")
        print(f"Nodos expandidos por busqueda en Anchura: {nodos_anchura}")
    else:
        print(" La busqueda en Anchura no encontro una solucion.")

if __name__ == "__main__":
    run_app()