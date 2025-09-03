#Tarea1
class Nodo:
    def __init__(self, valor):#Clase que representa un nodo del árbol, Cada nodo tiene un valor y referencias a sus hijos izquierdo y derecho.
        self.valor = valor
        self.izq = None
        self.der = None
class Arbol:
    def __init__(self):#Clase que representa un Árbol Binario de Búsqueda (BST),Contiene un nodo raíz.
        self.raiz = None

    # Método para insertar un valor en el árbol
    def insertar(self, valor):#Inserta un valor en el árbol binario de búsqueda,Si la raíz está vacía, crea el nodo raíz,En caso contrario, usa la función recursiva _insertar.
        if self.raiz is None:
            self.raiz = Nodo(valor)
        else:
            self._insertar(self.raiz, valor)

    def _insertar(self, actual, valor):#Inserta un valor en el árbol de manera recursiva, Si el valor es menor que el nodo actual, va a la izquierda.
        #Si el valor es mayor, va a la derecha,Si es igual, no se inserta (para evitar duplicados).
        if valor < actual.valor:  # va a la izquierda
            if actual.izq is None:
                actual.izq = Nodo(valor)
            else:
                self._insertar(actual.izq, valor)
        elif valor > actual.valor:  # va a la derecha
            if actual.der is None:
                actual.der = Nodo(valor)
            else:
                self._insertar(actual.der, valor)
        # Si es igual, no se hace nada (evita duplicados)

    # Método para imprimir el árbol (InOrden)
    def imprimirArbol(self):#Imprime el árbol en recorrido InOrden (Izquierda - Raíz - Derecha), Esto mostrará los valores en orden ascendente.
 
        self._imprimir(self.raiz)
        print()  # Salto de línea al final

    def _imprimir(self, actual):
        if actual is not None:
            self._imprimir(actual.izq)
            print(actual.valor, end=" ")
            self._imprimir(actual.der)

    # Método para buscar un valor en el árbol
    def buscarNodo(self, valor):#Busca un valor en el árbol binario de búsqueda,Retorna el nodo si se encuentra, en caso contrario None.
        return self._buscar(self.raiz, valor)

    def _buscar(self, actual, valor):
        if actual is None:
            return None
        if valor == actual.valor:
            return actual
        elif valor < actual.valor:
            return self._buscar(actual.izq, valor)
        else:
            return self._buscar(actual.der, valor)

    # Método para imprimir el árbol en PreOrden
    def imprimirPreOrden(self):#Recorre el árbol en PreOrden (Raíz - Izquierda - Derecha).
        self._preOrden(self.raiz)
        print()

    def _preOrden(self, actual):
        if actual is not None:
            print(actual.valor, end=" ")
            self._preOrden(actual.izq)
            self._preOrden(actual.der)

    # Método para imprimir el árbol en PostOrden
    def imprimirPostOrden(self):#Recorre el árbol en PostOrden (Izquierda - Derecha - Raíz).
        self._postOrden(self.raiz)
        print()

    def _postOrden(self, actual):
        if actual is not None:
            self._postOrden(actual.izq)
            self._postOrden(actual.der)
            print(actual.valor, end=" ")

#IMPLEMENTACION----------
arbol = Arbol()
arbol.insertar(50)
arbol.insertar(30)
arbol.insertar(70)
arbol.insertar(20)
arbol.insertar(40)
arbol.insertar(60)
arbol.insertar(80)

print("Recorrido InOrden:")
arbol.imprimirArbol()   # 20 30 40 50 60 70 80

print("Recorrido PreOrden:")
arbol.imprimirPreOrden()  # 50 30 20 40 70 60 80

print("Recorrido PostOrden:")
arbol.imprimirPostOrden()  # 20 40 30 60 80 70 50

print("Búsqueda del nodo 40:")
nodo = arbol.buscarNodo(40)
if nodo:
    print("Nodo encontrado con valor:", nodo.valor)
else:
    print("Nodo no encontrado")