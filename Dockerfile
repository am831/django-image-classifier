FROM python:3.11

ENV PYTHONUNBUFFERED 1

RUN mkdir /med_image_classify

WORKDIR /med_image_classify

ADD requirements.txt /med_image_classify/

RUN pip install -r requirements.txt

ADD . /med_image_classify/

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]