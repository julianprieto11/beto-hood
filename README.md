# BETO HOOD

Motor independiente de predicción, simulación y aprendizaje para apuestas de fútbol argentino.

## Principio central

Beto Hood aprende **fecha a fecha** sin mirar el futuro:

```
Fecha N
  ↓
histórico anterior a N
  ↓
features
  ↓
predicción
  ↓
10.000 simulaciones
  ↓
probabilidades
  ↓
resultado real de N
  ↓
errores
  ↓
memoria/calibración
  ↓
Fecha N+1
```

Una fecha nunca utiliza sus propios resultados para construir la predicción.

## Arquitectura

- `datos/`: datos crudos y datasets derivados.
- `modelos/`: configuración, features, predicción y aprendizaje.
- `simulacion/`: simulación Monte Carlo.
- `apuestas/`: probabilidades, cuotas justas y edge.
- `backtest/`: evaluación cronológica.
- `salidas/`: predicciones y simulaciones generadas localmente.

## Mercados iniciales

- goles
- corners
- tarjetas
- tiros totales
- tiros al arco
- big chances
- resultado 1X2
- BTTS
- líneas Over/Under

Los mercados de jugadores se incorporarán cuando conectemos el dataset de jugadores.

## Aprendizaje persistente

Se guardan localmente:

- `modelos/memoria/errores_historicos.csv`
- `modelos/memoria/calibracion_mercados.csv`
- `modelos/memoria/estado_modelo.json`

La memoria no se reinicia al ejecutar una nueva fecha.

## Ejecución

```bash
python ejecutar_fecha.py 1
python ejecutar_fecha.py 2
python ejecutar_fecha.py 12 --simulaciones 100000
```

Por defecto la competencia es Clausura.

Para backtest cronológico:

```bash
python backtest/backtest.py --hasta 10 --aprender
```

## Próximas capas

1. calibración más profunda por mercado y equipo;
2. modelos específicos para corners/tarjetas/tiros;
3. jugadores y mercados individuales;
4. cuotas reales;
5. detección de value/edge;
6. combinadas con correlación;
7. optimización de riesgo y backtest de rentabilidad.

Nunca se debe interpretar una probabilidad del modelo como garantía de ganancia.
