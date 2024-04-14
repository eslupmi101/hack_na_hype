FROM python:3.10.14

WORKDIR /app

COPY requirements.txt .

RUN pip install -r /app/requirements.txt --no-cache-dir 

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--workers", "2"]
