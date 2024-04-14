FROM python:3.10.14

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y libhdf5-dev

RUN pip install tensorflow==2.15

RUN pip install tqdm==4.66.2

RUN pip install scikit-learn==1.2.2

RUN pip install scipy==1.11.4

RUN pip install matplotlib==3.7.1

RUN pip install pandas==2.0.3

RUN pip install numpy==1.25.2

RUN pip install fastapi

RUN pip install uvicorn

RUN pip install python-dotenv

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--workers", "2"]
