import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
daily_visits = pd.read_csv('data/daily_visits_features.csv')
daily_visits['date'] = pd.to_datetime(daily_visits['date'])
prophet_data = daily_visits[['date', 'total_visits']].copy()
prophet_data.columns = ['ds', 'y']

# Праздники
holidays = pd.DataFrame({
    'holiday': 'holiday',
    'ds': pd.to_datetime([
        '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05',
        '2020-01-06', '2020-01-07', '2020-01-08', '2020-02-23', '2020-03-08',
        '2020-05-01', '2020-05-09', '2020-06-12', '2020-11-04',
        '2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05',
        '2021-01-06', '2021-01-07', '2021-01-08', '2021-02-23', '2021-03-08',
        '2021-05-01', '2021-05-09', '2021-06-12', '2021-11-04',
        '2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05',
        '2022-01-06', '2022-01-07', '2022-01-08', '2022-02-23', '2022-03-08',
        '2022-05-01', '2022-05-09', '2022-06-12', '2022-11-04',
        '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
        '2023-01-06', '2023-01-07', '2023-01-08', '2023-02-23', '2023-03-08',
        '2023-05-01', '2023-05-09', '2023-06-12', '2023-11-04',
        '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
        '2024-01-06', '2024-01-07', '2024-01-08', '2024-02-23', '2024-03-08',
        '2024-05-01', '2024-05-09', '2024-06-12', '2024-11-04',
        '2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05',
        '2025-01-06', '2025-01-07', '2025-01-08', '2025-02-23', '2025-03-08',
        '2025-05-01', '2025-05-09', '2025-06-12'
    ]),
    'lower_window': 0,
    'upper_window': 0
})

# Обучение модели
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05, holidays=holidays)
model.fit(prophet_data)

# Прогноз на весь период
future = model.make_future_dataframe(periods=0)
forecast = model.predict(future)

# Рисунок 3.1.2.2 — Компоненты модели
fig = model.plot_components(forecast, figsize=(15, 10))
plt.tight_layout()
plt.savefig('fig_3.1.2.2_prophet_components.png', dpi=300, bbox_inches='tight')
print("✓ Рисунок 3.1.2.2 сохранен как 'fig_3.1.2.2_prophet_components.png'")