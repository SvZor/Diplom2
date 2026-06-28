import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
daily_visits = pd.read_csv('data/daily_visits_features.csv')
daily_visits['date'] = pd.to_datetime(daily_visits['date'])

# Подготовка данных
prophet_data = daily_visits[['date', 'total_visits']].copy()
prophet_data.columns = ['ds', 'y']

# Разделение
train_end = pd.Timestamp('2024-12-31')
train_data = prophet_data[prophet_data['ds'] <= train_end]
test_data = prophet_data[prophet_data['ds'] > train_end]

print(f"Обучающая: {len(train_data)} дней")
print(f"Тестовая: {len(test_data)} дней")

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
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    changepoint_prior_scale=0.05,
    holidays=holidays
)
model.fit(train_data)
print("✓ Модель обучена")

# Прогноз
future = model.make_future_dataframe(periods=len(test_data))
forecast = model.predict(future)

# Метрики
y_true = test_data['y'].values
y_pred = forecast[forecast['ds'] > train_end]['yhat'].values

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100

print("\n" + "=" * 50)
print("МЕТРИКИ PROPHET")
print("=" * 50)
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# Рисунок
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(test_data['ds'], y_true, color='#2E86AB', linewidth=1.5, label='Фактические')
ax.plot(test_data['ds'], y_pred, color='#F18F01', linewidth=1.5, linestyle='--', label='Прогноз Prophet')
ax.set_title('Сравнение фактических и прогнозных значений (Prophet)', fontsize=14)
ax.set_xlabel('Дата')
ax.set_ylabel('Количество визитов')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_3.1.2.1_prophet_forecast.png', dpi=300, bbox_inches='tight')
print("✓ Рисунок сохранен")

# Сохранение метрик
with open('prophet_metrics.txt', 'w', encoding='utf-8') as f:
    f.write(f"MAE:  {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"MAPE: {mape:.2f}%\n")

print("✓ Метрики сохранены в 'prophet_metrics.txt'")
print("\nГОТОВО!")