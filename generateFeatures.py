##
## Code to generate a feature representation for VizWiz questions
##
## Author: Sarang Bhadsavle
## Additional credit: Brandon Dang (https://github.com/budang/INF385T/)

import skimage.io
import json
from pprint import pprint
import requests
# import matplotlib.pyplot as plt
from myAzureApiKeys import txt_analysis_key, cv_key

vision_base_url = "https://westcentralus.api.cognitive.microsoft.com/vision/v1.0/"
vision_analyze_url = vision_base_url + "analyze?"

# evaluate an image using the Microsoft Azure Cognitive Services Computer Vision API
def analyze_image(image_url):
    image = skimage.io.imread(image_url)
    # plt.imshow(image)
    # plt.axis("off")
    # plt.show()
    
    headers  = {'Ocp-Apim-Subscription-Key': cv_key }
    params   = {'visualFeatures': 'Adult,Categories,Description,Color,Faces,ImageType,Tags'}
    data     = {'url': image_url}
    
    response = requests.post(vision_analyze_url, headers=headers, params=params, json=data)
    response.raise_for_status()
    analysis = response.json()
    return analysis

def main():
	print("Starting main")
	# trainingData = json.load(open('data/train.json'))
	# pprint(trainingData[2])
	
	image_url = "https://cvc.ischool.utexas.edu/~dannag/VizWiz/Images/VizWiz_train_000000000000.jpg"
	data = analyze_image(image_url)
	pprint(data)


if __name__ == '__main__':
    main()