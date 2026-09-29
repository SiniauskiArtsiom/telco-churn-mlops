# Telco Churn Prediction — MLOps Service

End-to-end ML-сервис для прогнозирования оттока телеком-клиентов.

![CI](https://github.com/<username>/telco-churn-mlops/actions/workflows/ci.yml/badge.svg)

## Стек
- Python 3.10, Pandas, Scikit-learn, CatBoost
- FastAPI, Pydantic, Uvicorn
- Docker, docker-compose
- Prometheus-метрики
- pytest, ruff

## Быстрый старт

```bash
git clone https://github.com/<username>/telco-churn-mlops.git
cd telco-churn-mlops
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py
uvicorn app.main:app --reload
```

Открыть: http://localhost:8000/docs

## Docker

```bash
docker compose up --build
```

## API

| Метод | Путь         | Описание                  |
|-------|--------------|---------------------------|
| GET   | /health      | Статус сервиса            |
| GET   | /model-info  | Информация о модели       |
| POST  | /predict     | Предсказание вероятности  |
| GET   | /metrics     | Prometheus-метрики        |

## Тесты

```bash
pytest
ruff check .
```

## Метрики модели

См. `model/metrics.json`.
