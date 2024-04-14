FROM python:3.10

WORKDIR /app

RUN pip install poetry

COPY . .

RUN mkdir mediafiles

RUN poetry install

CMD ["poetry", "run", "gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "view.wsgi"]
