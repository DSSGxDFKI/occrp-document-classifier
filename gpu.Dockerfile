FROM nvcr.io/nvidia/tensorflow:22.06-tf2-py3

RUN apt-get update -y && apt-get install libgl1 -y
RUN pip install --upgrade pip
WORKDIR /app
COPY requirements_docker.txt .
RUN pip install -r requirements_docker.txt

ENV IN_DOCKER 1

COPY . .

ENTRYPOINT [ "python", "src/main.py"]