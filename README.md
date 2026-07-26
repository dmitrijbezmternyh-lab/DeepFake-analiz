 DeepFake-analiz

Модель для бинарной классификации лиц на реальные и сгенерированные изображения.

 Архитектура

- Backbone: EfficientNet‑B0 (обучен с нуля, без предобученных весов)
- Ключевая фича: High‑Pass Filter для выделения артефактов
- Метрика: F1‑score

  Результат

- F1: 0.75
- Recall: 0.78

Запуск

```bash
# Установка зависимостей
pip install -r requirements.txt

# Обучение модели
python train.py

# Предсказание на одном изображении
python predict.py --image path/to/image.jpg

Технологии
Python 3.10+

PyTorch, torchvision

NumPy, pandas, scikit‑learn

Pillow, OpenCV, matplotlib

Структура проекта
DeepFake-analiz/
├── README.md          # Описание проекта
├── requirements.txt   # Зависимости
├── config.py          # Конфигурация
├── train.py           # Обучение
├── predict.py         # Инференс
├── models/            # Архитектура модели
│   ├── __init__.py
│   ├── highpass_filter.py
│   └── deepfake_detector.py
├── dataset.py         # Загрузка данных
├── utils.py           # Вспомогательные функции
└── inference.ipynb    # Jupyter ноутбук для демонстрации

Автор
Безматерных Дмитрий
Студент УГНТУ (2 курс)
Выпускник Яндекс Лицея (специализация «Машинное обучение»)
