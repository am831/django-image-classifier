from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.core.files.storage import default_storage
import boto3
from botocore.exceptions import NoCredentialsError
import environ
from django.conf import settings
import pickle
import numpy as np
import os
import cv2
import urllib.request

modelBrain = pickle.load(open('imaging/detect_brain_tumor.pkl','rb'))
modelSkin = pickle.load(open('imaging/detect_melanoma.pkl','rb'))

def homepage(request):
    return render(request, 'homepage.html', { 'a' : 1})

def uploadBrain(request):
    return render(request, 'classifyBrain.html', {'a' : 1})

def uploadSkin(request):
    return render(request, 'classifySkin.html', {'a' : 1})

def predictImageBrain(request):
    """
    Get the user uploaded image from the request, saves it in the s3 bucket,
    and then uses the model to predict if the image has a tumor or not.
    """
    env = environ.Env()
    environ.Env.read_env()
    s3 = boto3.client('s3')
    AWS_STORAGE_BUCKET_NAME=env('AWS_STORAGE_BUCKET_NAME')
    if request.method == 'POST':
        file = request.FILES['filePath']
        folder = 'media/'
        objkey = f"{folder}{file.name}"
        try:
        # Upload the file to S3 bucket
            s3.upload_fileobj(file, AWS_STORAGE_BUCKET_NAME, objkey)
            file_url = s3.generate_presigned_url('get_object', Params={'Bucket': AWS_STORAGE_BUCKET_NAME, 'Key': objkey})
        except NoCredentialsError:
            return 'AWS credentials could not be found.'
        
    req=urllib.request.urlopen(file_url)
    image_data = req.read()
    np_arr = np.asarray(bytearray(image_data), dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    # Scale the feature vector to the desired number of features
    img = cv2.resize(image, (200,200))
    img = img.reshape(1,-1) / 255
    prediction = modelBrain.predict(img)
    if prediction == 0:
        prediction = "No tumor"
    elif prediction == 1:
        prediction = "Positive tumor"
    #copy file to new name with prediction, then delete the old file
    obj = s3.get_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=objkey)
    p = prediction.replace(" ", "") + "_" + file.name
    new_key = f"{folder}{p}"
    s3.copy_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=new_key, 
    CopySource={'Bucket': AWS_STORAGE_BUCKET_NAME, 'Key': objkey})
    s3.delete_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=objkey)
    file_url = s3.generate_presigned_url('get_object', Params={'Bucket': AWS_STORAGE_BUCKET_NAME, 'Key': new_key})

    context = {'filePath': file_url, 'prediction': prediction, 'filename': file.name}
    return render(request, 'classifyBrain.html', context)

def predictImageSkin(request):
    """
    Get the user uploaded image from the request, saves it in the s3 bucket,
    and then uses the model to predict if the image shows melanoma or not.
    """
    env = environ.Env()
    environ.Env.read_env()
    s3 = boto3.client('s3')
    AWS_STORAGE_BUCKET_NAME=env('AWS_STORAGE_BUCKET_NAME')
    if request.method == 'POST':
        file = request.FILES['filePath']
        folder = 'media/'
        objkey = f"{folder}{file.name}"
        try:
        # Upload the file to S3 bucket
            s3.upload_fileobj(file, AWS_STORAGE_BUCKET_NAME, objkey)
            file_url = s3.generate_presigned_url('get_object', Params={'Bucket': AWS_STORAGE_BUCKET_NAME, 'Key': objkey})
        except NoCredentialsError:
            return 'AWS credentials could not be found.'
        
    req=urllib.request.urlopen(file_url)
    image_data = req.read()
    np_arr = np.asarray(bytearray(image_data), dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    # Scale the feature vector to the desired number of features
    img = cv2.resize(image, (200,200))
    img = img.reshape(1,-1) / 255
    prediction = modelSkin.predict(img)
    if prediction == 0:
        prediction = "Benign"
    elif prediction == 1:
        prediction = "Malignant"
    #copy file to new name with prediction, then delete the old file
    obj = s3.get_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=objkey)
    p = prediction + "_" + file.name
    new_key = f"{folder}{p}"
    s3.copy_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=new_key, 
    CopySource={'Bucket': AWS_STORAGE_BUCKET_NAME, 'Key': objkey})
    s3.delete_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=objkey)
    file_url = s3.generate_presigned_url('get_object', Params={'Bucket': AWS_STORAGE_BUCKET_NAME, 'Key': new_key})

    context = {'filePath': file_url, 'prediction': prediction, 'filename': file.name}
    return render(request, 'classifySkin.html', context)

def viewDatabase(request):
    sess = boto3.Session()
    env = environ.Env()
    environ.Env.read_env()
    AWS_STORAGE_BUCKET_NAME=env('AWS_STORAGE_BUCKET_NAME')
    AWS_ACCESS_KEY_ID=env('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY=env('AWS_SECRET_ACCESS_KEY')
    AWS_S3_REGION_NAME=env('AWS_S3_REGION_NAME')
    s3 = boto3.resource('s3',
                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                        region_name=AWS_S3_REGION_NAME)
    bucket = s3.Bucket(AWS_STORAGE_BUCKET_NAME)
    folder = "media/"
    file_list=[]
    s3client = boto3.client('s3')
    for obj in bucket.objects.filter(Prefix=folder):
            file_url = s3client.generate_presigned_url('get_object', Params={'Bucket': AWS_STORAGE_BUCKET_NAME, 'Key': obj.key})
            file_list.append(file_url)

    context={'listOfImagesPath':file_list}
    return render(request, 'viewDB.html', context)
