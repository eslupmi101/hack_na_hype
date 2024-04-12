FROM python:3.10.12

WORKDIR /app

RUN pip install poetry

COPY . .

RUN poetry install

CMD ["poetry", "run", "gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "view.wsgi"]
