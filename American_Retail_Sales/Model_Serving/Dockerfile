FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY ./app /app

COPY ./Models /Models
COPY ./Model_Training  /Model_Training
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "/gunicorn_conf.py", "main:app"]
