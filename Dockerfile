FROM python:3.8


COPY . /app

RUN apt-get update -y
# not sure if all of the following packages are necessary. introduced to fix some bug which arised when running the container
# maybe take it out again to speed up the conter image creation process
RUN apt-get install ffmpeg libsm6 libxext6 libgl1-mesa-glx -y
RUN apt-get install libreoffice vim tree -y
RUN python -m pip install --upgrade pip
WORKDIR /app/
RUN pip install -r /app/requirements.txt
RUN pip install -e /app
RUN sed -i 's/DEFAULT@SECLEVEL=2/DEFAULT@SECLEVEL=1/' /etc/ssl/openssl.cnf


WORKDIR /app/src/
