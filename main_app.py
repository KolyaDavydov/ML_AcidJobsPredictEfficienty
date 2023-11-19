import numpy as np
import pandas as pd

# Загрузка датасета
path_data = 'datasets/data_oil_well_2022.xlsx'
df = pd.read_excel(path_data)

# Удаляем строки где в 18 столбце есть NULL
df.dropna(subset=['Прирост нефти от кислотных обработок, тн'], inplace=True)

# ПРеобразуем столбец ГРП в котегориальный признак (1 - есть ГРП, 2 - нет ГРП)
df['Наличие ГРП'] = df['Наличие ГРП'].fillna(0)
df.loc[df['Наличие ГРП'] != 0, 'Наличие ГРП'] = 1
df['Наличие ГРП'] = df['Наличие ГРП'].astype(int)

# Удалим ненужные столбцы которые не могут сказываться на результате
df = df.drop('Проницаемость, мд', axis=1)
df = df.drop('№ скважины', axis=1)
df = df.drop('Месторождение', axis=1)
df = df.drop('Пласт', axis=1)
df = df.drop('Дата ввода скважины в эксплуатацию', axis=1)

# Преобразуем некоторые данные в категориальные
features = ['Тип скважины', 'Тип коллектора']
for feature in features:
  df[feature] = df[feature].astype('category')

# Добавялем столбец `Эффективность` именно этот параметр необходимо будет предстказывать.
# Эффективность=true если прирост дебита нефти после кислотных ОПЗ будет больше 3 тн/сут, и на оборот.
# если эффективность = 1, то успешно
df['Эффективность'] = np.where(df['Прирост нефти от кислотных обработок, тн']>3, 1, 0)

# Разделяем данные на обучающий и тестовый наборы
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = df.drop('Эффективность', axis=1)
X = X.drop('Прирост нефти от кислотных обработок, тн', axis=1)
y = df['Эффективность']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 100)

# Обучаем модель
from catboost import CatBoostClassifier
cat_model = CatBoostClassifier(iterations=100, depth=10, learning_rate=0.1, cat_features=features)
cat_model.fit(X_train, y_train)

# Делаем предсказание на тестовом наборе данных
y_pred = cat_model.predict(X_test)

# Оцениваем точность модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy}")

# Сохранение модели
cat_model.save_model('model/efficiency_acid_jobs.cbm')

def predict(well):
  # получаем список названий столбцов которые нужны для предсказания
  col_name = [column for column in df]
  col_name.pop()
  col_name.pop()

# Такой список должна принять функция
  # well = [well,
  #           type_form,
  #           power_form,
  #           begin_pressure,
  #           current_pressure,
  #           current_q_oil,
  #           current_q_liq,
  #           percent_water,
  #           productivity,
  #           skin,
  #           temperature,
  #           viscosity,
  #           frac]
  values = np.array([well])
  data_pred = pd.DataFrame(values, columns=col_name)

  predict = cat_model.predict(data_pred)

  result = ''
  match predict:
    case 0:
      result='НЕ рекомендуется'
    case 1:
      result='Рекомендуется'

  return result



# Пример функции:
"""
well_1 = ["ВЕРT",
            "Карбонат-известняк",
            4.0,
            345,
            300,
            1.7,
            3.0,
            33,
            0.01,
            154.2,
            59,
            1.7,
            0]

print(predict(well_1))
"""