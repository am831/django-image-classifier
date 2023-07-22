# Django Medical Image Classifier
A Django webapp that uses machine learning models to classify images of melanoma and brain MRI scans as benign vs malignant. The user can upload images which are stored in an AWS S3 bucket.
![homepage](https://github.com/am831/django-image-classifier/assets/59581465/6f1a0ef4-a1b2-452d-b5ee-36ace14b641f)


# Image Classification with Scikit Learn
Two different machine learning models were trained using datasets downloaded from Kaggle. One model can determine if a brain MRI image contains a tumor or not, the other can determine if a skin abnormality is melanoma or not. Models were trained with Scikit-Learn using the support vector machine (SVM) algorithm.

- Brain tumor classification dataset can be downloaded [here](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri?resource=download)
- Melanoma classification dataset can be downloaded [here](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images?resource=download)

# Machine Learning Model Performance
BrainMRI ML model:
- Training score: 0.9812717770034843
- Testing score: 0.9355400696864111
- Accuracy score: 0.8401015228426396

Melanoma ML model:
- Training score: 0.9068193649141072
- Testing score: 0.8870380010411244
- Accuracy score: 0.897

# Installation
- Download the docker image alisha831maddy/med_image_classify (4.38 GB)
- Run the image with the command "docker run -p 8000:8000 alisha831maddy/med_image_classify"
- The server will start at http://0.0.0.0:8000, but you need to navigate to http://localhost:8000
- Download images to work with the ML models, see previous section for links to image datasets

# Usage
At the homepage shown above, click on the ML model you want to use. You'll be directed to a page where you can upload images for classification.
![braintumor](https://github.com/am831/django-image-classifier/assets/59581465/c60a2d46-f7f6-438d-ab21-36e543e1741f)
![melanoma](https://github.com/am831/django-image-classifier/assets/59581465/e129fdac-47b3-4b75-bbf7-9526cb5bfbfe)
Navigate to View Gallery to see previously uploaded images that are stored in an AWS S3 bucket.
![githubdemo](https://github.com/am831/django-image-classifier/assets/59581465/6d5066aa-c746-4742-ac22-6634f188f0c4)

# Built With

- **Web Development:** <br>
Django | HTML | CSS 

- **Python 3.11 / Machine Learning** <br>
Scikit-Learn | Scikit-image | Numpy

- **Database** <br>
AWS S3

- **Deployment:** <br>
Docker
