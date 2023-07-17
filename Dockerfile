FROM python:3.11

ENV PYTHONUNBUFFERED 1

RUN mkdir /med_image_classify

WORKDIR /med_image_classify

ADD . /med_image_classify/

RUN pip install -r requirements.txt

