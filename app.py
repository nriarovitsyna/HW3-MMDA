import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn import metrics

# Ответ на вопрос 4.5:
# Для модели используется st.cache_resource, а для таблиц - st.cache_data, так как:
# st.cache_resource хранит один общий объект (модель, соединение) для всех пользователей
# и всех перезапусков скрипта. Объект не копируется, это один и тот же экземпляр. А нам как раз и 
# нужно загрузить модель один раз и использовать как есть.
# st.cache_data хранит данные (таблицы, массивы, результаты вычислений) и при каждом
# вызове возвращает копию закэшированного значения, поэтому изменение таблицы в одном
# месте не портит кэш.
# Если вызвать joblib.load без декоратора, он будет выполняться при каждом перезапуске
# скрипта (движение слайдера, клик, загрузка файла), модель будет каждый раз читаться
# с диска и создаваться заново, и приложение станет работать медленнее.

APP_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


model = load_model()
config = json.loads((APP_DIR / 'config.json').read_text())
test = load_test()

features = [c for c in test.columns if c != 'target']   # 12 признаков в порядке обучения

st.title('Рекомендации вин')

# Сайдбар: порог (по умолчанию из config.json)
threshold = st.sidebar.slider('Порог', 0.0, 1.0, float(config['threshold']), 0.01)

tab_single, tab_metrics, tab_batch, tab_coef = st.tabs(['Оценка одного вина', 'Модельные метрики', 'Оценка нескольких вин', 'Влияние признаков'])

# Вкладка 1: Оценка одного вина
with tab_single:
    st.subheader('Введите характеристики вина')
    values = {}
    cols = st.columns(3)
    i = 0
    # Числовые признаки: диапазон и значение по умолчанию берем из test.csv
    for feature in features:
        if feature == 'is_red':
            continue
        vmin = float(test[feature].min())
        vmax = float(test[feature].max())
        vmed = float(test[feature].median())
        with cols[i % 3]:
            values[feature] = st.number_input(
                feature, min_value=vmin, max_value=vmax, value=vmed,
                step=(vmax - vmin) / 100, format='%.4f')
        i += 1

    # Переключатель «красное / белое»;
    # По умолчанию то, что чаще встречается в тесте
    wine_type = st.radio('Тип вина', ['Белое', 'Красное'],
                         index=int(test['is_red'].median()), horizontal=True)
    values['is_red'] = 1 if wine_type == 'Красное' else 0

    x = pd.DataFrame([values])[features]
    proba = model.predict_proba(x)[0, 1]

    st.metric('Вероятность, что вино хорошее', f'{proba:.1%}')
    if proba >= threshold:
        st.success(f'Решение модели: рекомендовать (вероятность {proba:.3f} ≥ порог {threshold:.2f})')
    else:
        st.warning(f'Решение модели: не рекомендовать (вероятность {proba:.3f} < порог {threshold:.2f})')

# Вкладка 2: Модельные метрики
with tab_metrics:
    st.subheader(f'Метрики на тестовой выборке при пороге {threshold:.2f}')

    y_test = test['target']
    scores = test_proba(model, test[features])
    y_pred = (scores >= threshold).astype(int)

    # матрица ошибок
    conf_matrix = metrics.confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = conf_matrix.ravel()

    # precision, recall, F1
    precision = metrics.precision_score(y_test, y_pred, zero_division=0)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, zero_division=0)

    col_cm, col_metrics = st.columns(2)
    with col_cm:
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
        metrics.ConfusionMatrixDisplay(
            conf_matrix, display_labels=['не хорошее', 'хорошее']).plot(ax=ax_cm, colorbar=False)
        ax_cm.set_title('Матрица ошибок')
        st.pyplot(fig_cm)
        plt.close(fig_cm)
    with col_metrics:
        st.metric('Precision', f'{precision:.3f}')
        st.metric('Recall', f'{recall:.3f}')
        st.metric('F1', f'{f1:.3f}')

    # ROC- и PR-кривые с точкой текущего порога
    fpr, tpr, _ = metrics.roc_curve(y_test, scores)
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(y_test, scores)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(fpr, tpr, label='ROC-кривая')
    ax[0].plot([0, 1], [0, 1], 'k--', label='случайная модель')
    ax[0].scatter(fp / (fp + tn), recall, color='red', label='текущий порог')
    ax[0].set_xlabel('FPR')
    ax[0].set_ylabel('TPR (recall)')
    ax[0].set_title('ROC-кривая')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(recall_curve, precision_curve, label='PR-кривая')
    ax[1].scatter(recall, precision, color='red', label='текущий порог')
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].set_title('PR-кривая')
    ax[1].legend()
    ax[1].grid(True)

    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        '**Что это значит для подборки.** '
        'Если порог увеличить, модель будет рекомендовать меньше вин: подборка становится короче, '
        'но в ней реже встречаются плохие вина (так как precision растет), '
        'но больше хороших вин остается пропущенными (recall падает). '
        'Если порог уменьшить, подборка становится шире и находит больше хороших вин '
        '(recall растет), но в нее чаще станут попадать плохие вина (precision падает).')
    
# Вкладка 3: Пакетное предсказание
with tab_batch:
    st.subheader('Оценка нескольких вин')
    st.write(f'Порог: {threshold:.2f} (меняется ползунком в сайдбаре)')

    # Загрузка CSV-файла пользователем
    uploaded = st.file_uploader(
        'Загрузите CSV-файл с признаками вин (без столбца target)', type='csv')

    if uploaded is not None:
        # Если файл не читается как CSV, показываем ошибку и останавливаем вкладку
        try:
            data = pd.read_csv(uploaded)
        except Exception:
            st.error('Не удалось прочитать файл. Проверьте, что это корректный CSV.')
            st.stop()

        # Проверяем, что в файле есть все 12 признаков, на которых обучалась модель
        missing = [c for c in features if c not in data.columns]
        if missing:
            st.error('В файле не хватает столбцов: ' + ', '.join(missing))
        else:
            try:
                proba_new = model.predict_proba(data[features])[:, 1]
            except ValueError:
                st.error('Не удалось выполнить предсказание: в признаках есть пропуски '
                         'или нечисловые значения.')
                st.stop()

            # Добавляем к исходной таблице вероятность и решение при текущем пороге
            result = data.copy()
            result['probability'] = proba_new
            result['decision'] = ['рекомендовать' if p >= threshold else 'не рекомендовать'
                                  for p in proba_new]

            st.dataframe(result)
            st.download_button(
                'Скачать результат (CSV)',
                data=result.to_csv(index=False).encode('utf-8'),
                file_name='predictions.csv',
                mime='text/csv')
    
# Вкладка 4: Влияние признаков
with tab_coef:
    st.subheader('Влияние признаков')

    coefs = pd.Series(model[-1].coef_[0], index=features).sort_values()

    # Горизонтальная столбчатая диаграмма: красные против хорошего вина, зеленые за
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['tab:red' if c < 0 else 'tab:green' for c in coefs]
    ax.barh(coefs.index, coefs.values, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Коэффициент (признаки масштабированы)')
    ax.set_title('Коэффициенты логистической регрессии')
    ax.grid(True, axis='x')
    st.pyplot(fig)
    plt.close(fig)

    # Три признака с самыми большими положительными и отрицательными коэффициентами
    top_for = coefs[coefs > 0].sort_values(ascending=False).head(3)
    top_against = coefs[coefs < 0].sort_values().head(3)

    st.markdown('**Сильнее всего «голосуют» за хорошее вино:** '
                + ', '.join(f'{n} ({v:+.2f})' for n, v in top_for.items()))
    st.markdown('**Сильнее всего «голосуют» против:** '
                + ', '.join(f'{n} ({v:+.2f})' for n, v in top_against.items()))
    
