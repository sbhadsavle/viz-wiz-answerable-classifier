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
from pandas import DataFrame
from IPython.display import display
import cv2
import nltk
from myAzureApiKeys import txt_analysis_key, cv_key

vision_base_url = "https://westcentralus.api.cognitive.microsoft.com/vision/v1.0/"
vision_analyze_url = vision_base_url + "analyze?"

# Followed technique found here: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
# and https://www.pyimagesearch.com/2015/03/02/convert-url-to-image-with-python-and-opencv/
def detect_blur(image):
    correctedImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(correctedImage, cv2.COLOR_BGR2GRAY)
    varLap = cv2.Laplacian(gray, cv2.CV_64F).var()
    return varLap

# conduct POS tagging and return counts of POS types
def pos_tag_text(text):
    text = nltk.word_tokenize(text)
    tags = nltk.pos_tag(text)
    frequencies = nltk.FreqDist(tag for (word, tag) in tags)
    
    tag_counts = {
        '.': 0,
        'ADJ': 0,
        'ADP': 0,
        'ADV': 0,
        'CC': 0,
        'CONJ': 0,
        'DET': 0,
        'IN': 0,
        'JJ': 0,
        'NN': 0,
        'NOUN': 0,
        'NUM': 0,
        'PRON': 0,
        'PRT': 0,
        'RB': 0,
        'VERB': 0,
        'X': 0
    }
    
    for tag, count in frequencies.most_common():
        tag_counts[tag] = count
        
    return tag_counts

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
    # image = skimage.io.imread(image_url)
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


'''
(ex)

feature_set = {
"one" : [0,1,2,3,4],
"two" : [7,3,5,3,5],
"three" : [838,465,23,56,5]
}

    one three   two
0   0   838      7
1   1   465      3
2   2   23       5
3   3   56       3
4   4   5        5

'''
def main():
    print("Starting main")
    # trainingData = json.load(open('data/train.json'))
    # pprint(trainingData[2])
    
    # image_url = "https://cvc.ischool.utexas.edu/~dannag/VizWiz/Images/VizWiz_train_000000000003.jpg"
    # image = skimage.io.imread(image_url)
    # imgAnalysis = analyze_image(image_url)
    # pprint(imgAnalysis)

    # print("\n --- FEATURES --- \n")

    # features = extract_features(imgAnalysis)
    # pprint(features)

    # print("\n --- DATA FRAME --- \n")

    # df = DataFrame(data=features, index=[0])
    # display(df)

    # print("\n\nBlurriness index: " + str(detect_blur(image)))

    # nltk.download('tagsets')
    text = 'And now for something completely different'
    pos_counts = pos_tag_text(text)
    pprint(pos_counts)

if __name__ == '__main__':
    main()