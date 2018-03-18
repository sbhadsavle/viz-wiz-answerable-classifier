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

# improved feature extraction using one-hot encodings, feature binarization, and improved feature selection
def extract_features(data):
    return {
        "is_image_format_jpg"                : int(data["metadata"]["format"] == "Jpeg"),
        "is_image_format_png"                : int(data["metadata"]["format"] == "Png"),
        "image_height"                       : data["metadata"]["height"],
        "image_width"                        : data["metadata"]["height"],
        "clip_art_type"                      : data["imageType"]["clipArtType"],
        "line_drawing_type"                  : data["imageType"]["lineDrawingType"],
        "is_black_and_white"                 : int(data["color"]["isBwImg"]),
        "is_adult_content"                   : int(data["adult"]["isAdultContent"]),
        "adult_score"                        : data["adult"]["adultScore"],
        "is_racy"                            : int(data["adult"]["isRacyContent"]),
        "racy_score"                         : data["adult"]["racyScore"],
        "has_faces"                          : int(len(data["faces"])),
        "num_faces"                          : len(data["faces"]),
        "is_dominant_color_background_black" : int(data["color"]["dominantColorBackground"] == "Black"),
        "is_dominant_color_foreground_black" : int(data["color"]["dominantColorForeground"] == "Black")
    }

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
    imgAnalysis = analyze_image(image_url)
    pprint(imgAnalysis)

    print("\n --- FEATURES --- \n")

    features = extract_features(imgAnalysis)
    pprint(features)

if __name__ == '__main__':
    main()