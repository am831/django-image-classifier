# Django Medical Image Classifier
A Django webapp that uses machine learning models to classify images of melanoma and brain MRI scans as benign vs malignant. The user can upload images which are stored in an aws s3 bucket.
![homepage](https://github.com/am831/django-image-classifier/assets/59581465/90fba4bf-143e-4ec5-96ca-bd1c9a5da5e7)

# Image Classification with Scikit Learn
Two different machine learning models were trained using datasets downloaded from Kaggle. One model can determine if a brain MRI image contains a tumor or not, the other can determine if a skin abnormality is melanoma or not. Neural networks were built with Scikit-Learn.

- Brain tumor classification dataset can be downloaded [here](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri?resource=download)
- Melanoma classification dataset can be downloaded [here](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images?resource=download)

# Installation

# Usage
At the homepage shown above, click on the ML model you want to use. You'll be directed to a page where you can upload images for classification.
![braintumor](https://github.com/am831/django-image-classifier/assets/59581465/c7f4204e-feef-4791-991d-6eee9ae64738)
![melanoma](https://github.com/am831/django-image-classifier/assets/59581465/e120006a-e447-4f97-9c07-256677279c8f)

If you click on View Gallery, you'll be directed to a page where you can view previously uploaded images.

# Built With

- Web Development:
Django | HTML | CSS

- Python 3.11 / Nueral Network
Scikit-Learn | Numpy

- Database
AWS S3
