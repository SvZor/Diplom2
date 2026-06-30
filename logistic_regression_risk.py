import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Цветовая схема
COLORS = {
    'primary': '#2E86AB',
    'accent': '#F18F01',
    'positive': '#2E7D32',
}

print("=" * 60)
print("ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ ДЛЯ ОЦЕНКИ РИСКОВ")
print("=" * 60)

# 1. Загрузка данных
df = pd.read_csv('data/patient_features.csv')
print(f"Загружено пациентов: {len(df)}")

# 2. Преобразование категориальных признаков
df['gender'] = df['gender'].map({'M': 1, 'F': 0})
le = LabelEncoder()
df['patient_type_encoded'] = le.fit_transform(df['patient_type'])

# 3. Отбор признаков (убрали is_senior)
feature_cols = [
    'gender', 'age', 'patient_type_encoded',
    'visits_6m', 'visits_12m', 'cost_6m', 'cost_12m', 
    'avg_cost_6m', 'avg_cost_12m', 'specialist_share'
] + [col for col in df.columns if col.startswith('visits_') and col not in ['visits_6m', 'visits_12m']]

feature_cols = list(set(feature_cols))
X = df[feature_cols]
y = df['high_risk']

print(f"Количество признаков: {len(feature_cols)}")
print(f"Доля пациентов с высоким риском: {y.mean() * 100:.1f}%")

# 4. Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Разделение
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Обучающая: {len(X_train)} пациентов")
print(f"Тестовая: {len(X_test)} пациентов")

# 6. Обучение
model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)
print("\n✓ Модель обучена")

# 7. Оценка
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 50)
print("МЕТРИКИ КАЧЕСТВА (ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ)")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=['Низкий риск', 'Высокий риск']))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {roc_auc:.3f}")

# 8. Коэффициенты
coef_df = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': model.coef_[0]
})
coef_df['odds_ratio'] = np.exp(coef_df['coefficient'])
coef_df = coef_df.sort_values('coefficient', key=abs, ascending=False)

print("\n" + "=" * 50)
print("ТОП-5 ФАКТОРОВ РИСКА")
print("=" * 50)
print(coef_df.head(5).to_string(index=False))

# 9. ROC-кривая
fig, ax = plt.subplots(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax.plot(fpr, tpr, color=COLORS['primary'], linewidth=2, label=f'ROC-AUC = {roc_auc:.3f}')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC-кривая (Логистическая регрессия)')
ax.legend()
plt.tight_layout()
plt.savefig('fig_3.2.2.1_logreg_roc.png', dpi=300, bbox_inches='tight')
print("✓ Рисунок 3.2.2.1 сохранен")

# 10. Матрица ошибок
fig2, ax2 = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Низкий риск', 'Высокий риск'],
            yticklabels=['Низкий риск', 'Высокий риск'])
ax2.set_title('Матрица ошибок (Логистическая регрессия)')
plt.tight_layout()
plt.savefig('fig_3.2.2.2_logreg_cm.png', dpi=300, bbox_inches='tight')
print("✓ Рисунок 3.2.2.2 сохранен")

print("\nГОТОВО!")