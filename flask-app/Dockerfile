FROM python:3.10

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

CMD ["python", "server.py"]