import os

from kaggle.api.kaggle_api_extended import KaggleApi

dataset_name = 'aunanya875/suicidal-tweet-detection-dataset'

api = KaggleApi()
api.authenticate()

api.dataset_download_files(dataset_name, path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"), unzip=True)
