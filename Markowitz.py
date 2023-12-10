import numpy as np
from scipy.optimize import minimize

def obtener_datos_usuario():
    n = int(input("Ingrese la cantidad de activos en la cartera: "))
    rendimientos_esperados = []
    matriz_covarianza = np.zeros((n, n))

    for i in range(n):
        r = float(input(f"Ingrese el rendimiento esperado del activo {i+1}: "))
        rendimientos_esperados.append(r)

    for i in range(n):
        for j in range(i, n):
            if i == j:
                var = float(input(f"Ingrese la varianza del activo {i+1}: "))
                matriz_covarianza[i][j] = var
            else:
                cov = float(input(f"Ingrese la covarianza entre el activo {i+1} y el activo {j+1}: "))
                matriz_covarianza[i][j] = matriz_covarianza[j][i] = cov

    return np.array(rendimientos_esperados), matriz_covarianza

def funcion_objetivo(pesos, rendimientos_esperados, matriz_covarianza, riesgo_aversion):
    portafolio_retorno = np.dot(pesos, rendimientos_esperados)
    portafolio_varianza = np.dot(pesos.T, np.dot(matriz_covarianza, pesos))
    return -portafolio_retorno + riesgo_aversion * portafolio_varianza

def optimizar_cartera(rendimientos_esperados, matriz_covarianza, riesgo_aversion):
    n = len(rendimientos_esperados)
    restriccion = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    limites = tuple((0, 1) for _ in range(n))
    resultado = minimize(funcion_objetivo, np.ones(n)/n, args=(rendimientos_esperados, matriz_covarianza, riesgo_aversion),
                         constraints=restriccion, bounds=limites)
    return resultado.x

def main():
    rendimientos_esperados, matriz_covarianza = obtener_datos_usuario()
    riesgo_aversion = float(input("Ingrese su nivel de aversión al riesgo (mayor número indica mayor aversión): "))
    pesos_optimos = optimizar_cartera(rendimientos_esperados, matriz_covarianza, riesgo_aversion)

    # Convertir los pesos a porcentajes y formatear para una mejor lectura
    pesos_porcentaje = [f"{peso * 100:.2f}%" for peso in pesos_optimos]
    print("Distribución óptima de la cartera:")
    for i, peso in enumerate(pesos_porcentaje):
        print(f"Activo {i + 1}: {peso}")

if __name__ == "__main__":
    main()
