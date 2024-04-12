# Решение команды На Хайпе

### Запуск проекта через Docker

Создать .env, переменные окружения
Нужно создать .env файл и переместить туда содержимое файла .env.dist
```
cp .env.dist .env
```

```
docker-compose -f docker-compose.yml up -d
```

### Веб интерфейс
```
localhost:8000
```

### AI API
```
localhost:8001
```

### Документация AI API
```
localhost:8001/docs
```
