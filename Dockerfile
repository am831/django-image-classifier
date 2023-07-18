FROM python:3.11

ENV PYTHONUNBUFFERED 1

RUN mkdir /med_image_classify

WORKDIR /med_image_classify

ADD . /med_image_classify/

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["gunicorn", "--bind", ":8000", "--workers", "3", "mysite.wsgi"]