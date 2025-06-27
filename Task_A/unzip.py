import zipfile
import os

zip_path = 'best_coatnet_gender_model.zip'
extract_dir = ''

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
