import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Цветовая схема
COLORS = {
    'primary': '#2E86AB',
    'accent': '#F18F01',
    'negative': '#D32F2F',
    'positive': '#2E7D32',
    'gray': '#808080',
    'light_gray': '#D3D3D3',
}

print("=" * 60)
print("XGBOOST ДЛЯ ПРОГНОЗИРОВАНИЯ ВРЕМЕННЫХ РЯДОВ")
print("=" * 60)

# Загрузка данных
df = pd.read_csv('data/daily_visits_features.csv')
df['date'] = pd.to_datetime(df['date'])

# Удаляем первые 30 дней (где есть NaN из-за лагов)
df = df.dropna().reset_index(drop=True)
print(f"Данных после удаления пропусков: {len(df)} дней")

# Признаки и целевая переменная
feature_cols = [col for col in df.columns if col not in ['date', 'total_visits', 'total_revenue']]
X = df[feature_cols]
y = df['total_visits']

print(f"Количество признаков: {len(feature_cols)}")

# Разделение на обучающую и тестовую (с сохранением порядка)
split_date = '2024-12-31'
train_idx = df['date'] <= split_date
test_idx = df['date'] > split_date

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Обучающая выборка: {len(X_train)} дней")
print(f"Тестовая выборка: {len(X_test)} дней")

# Обучение XGBoost
print("\nОбучение XGBoost...")
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
model.fit(X_train, y_train)
print("✓ Модель обучена")

# Прогноз
y_pred = model.predict(X_test)

# Метрики
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100

print("\n" + "=" * 50)
print("МЕТРИКИ XGBOOST")
print("=" * 50)
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# Сохранение метрик
with open('xgboost_metrics.txt', 'w', encoding='utf-8') as f:
    f.write(f"MAE:  {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"MAPE: {mape:.2f}%\n")

# Важность признаков
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "=" * 50)
print("ТОП-10 ВАЖНЫХ ПРИЗНАКОВ")
print("=" * 50)
print(importance.head(10).to_string(index=False))

# Рисунок 3.1.3.1 — Сравнение фактических и прогнозных значений (XGBoost)
test_dates = df[test_idx]['date'].values
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(test_dates, y_test.values, color=COLORS['primary'], linewidth=1.5, label='Фактические')
ax.plot(test_dates, y_pred, color=COLORS['accent'], linewidth=1.5, linestyle='--', label='Прогноз XGBoost')
ax.set_title('Сравнение фактических и прогнозных значений (XGBoost)', fontsize=14)
ax.set_xlabel('Дата')
ax.set_ylabel('Количество визитов')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_3.1.3.1_xgboost_forecast.png', dpi=300, bbox_inches='tight')
print("✓ Рисунок 3.1.3.1 сохранен как 'fig_3.1.3.1_xgboost_forecast.png'")

# Рисунок 3.1.3.2 — Важность признаков
fig2, ax2 = plt.subplots(figsize=(10, 6))
top_features = importance.head(10)
ax2.barh(top_features['feature'], top_features['importance'], color=COLORS['primary'])
ax2.set_title('Топ-10 важных признаков (XGBoost)', fontsize=14)
ax2.set_xlabel('Важность')
ax2.set_ylabel('Признак')
ax2.invert_yaxis()
plt.tight_layout()
plt.savefig('fig_3.1.3.2_xgboost_importance.png', dpi=300, bbox_inches='tight')
print("✓ Рисунок 3.1.3.2 сохранен как 'fig_3.1.3.2_xgboost_importance.png'")

print("\nГОТОВО!")