# Решение команды На Хайпе

В папке Colab лежат файлы ноутбуков с моделями и пояснениями к графикам

### Описание проекта как сервиса

Реализован веб интерфес Django localhost:8000.
Через него можно загрузить train.csv и скачать таблицу csv с результатом
Также можно просмотреть демо данные

![alt text](https://github.com/eslupmi101/hack_na_hype/blob/ba258c27243eedcd68a40a71031f7f252d107fff/%D1%81%D0%BA%D1%80%D0%B8%D0%BD.png)

### Стэк используемых основных технологий:

| программа                     | версия |
|-------------------------------|--------|
| Python                        | 3.10.12|
| Django                        | 3.2    |
| gunicorn                      | 21.2.0 |
| Docker                        | 23.0.2 |
| fastapi                       | 0.25.0 |
| uvicorn                       | 0.95.0 | 

### Запуск проекта через localhost

Создать .env, переменные окружения
Нужно создать .env файл и переместить туда содержимое файла .env.dist
```
cp .env.dist .env
```

Поменять переменную AI_API_URL
```
AI_API_URL="http://localhost:8001" 
```

Войти в папку view
```
cd src/view
```

Установить зависимости
```
poetry install --no-root
```

Запустить сервер
```
poetry run python manage.py runserver
```

Открыть второй терминал
```
cd src/ai
```

Создать venv
```
python -m venv venv
```

Активировать venv
```
source venv/bin/activate
```
или
```
source venv/Scripts/Activate
```

Запустить сервер
```
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Зайти через браузер по порту localhost:8000.
Отправить файл train.cvs.
Дождаться демоданных.
Когда появятся демоданные можно будет скачать файл с результатом

### Запуск проекта через Docker

Создать .env, переменные окружения
Нужно создать .env файл и переместить туда содержимое файла .env.dist
```
cp .env.dist .env
```

Запустить докер
```
docker-compose -f docker-compose.yml up -d
```


### UPD Docker
```
Во время ожидания результата сайт может упасть, но после перезагрузки, когда файл будет готов, то скачивание файла станет доступным (демо данные не отображаются)
```

### Мануальный запуск

Войдите в директорию
```
cd src/ai
```

Скопируйте файл train.csv в папку mediafiles
```
src/ai/mediafiles/train.csv
```

Создать venv
```
python -m venv venv
```

Активировать venv
```
source venv/bin/activate
```
или
```
source venv/Scripts/Activate
```

Запустите файл manual.py
```
python manual.py
```

Найдите результаты в mefiafiles/result.csv
```
src/ai/mediafiles/result.csv
```