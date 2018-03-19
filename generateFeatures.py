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

text_analytics_base_url = "https://westcentralus.api.cognitive.microsoft.com/text/analytics/v2.0/"
sentiment_api_url = text_analytics_base_url + "sentiment"
key_phrase_api_url = text_analytics_base_url + "keyPhrases"

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

def get_sentiments(question):
    documents = []
    documents.append({
        'id': 0,
        'text': question
    })

    param = {'documents': documents}
    headers   = {"Ocp-Apim-Subscription-Key": txt_analysis_key}
    response  = requests.post(sentiment_api_url, headers=headers, json=param)
    sentimentJson = response.json()
    # pprint(sentiment)
    #print(sentimentJson['documents'][0]['score'])
    return sentimentJson['documents'][0]['score']

def get_num_key_phrases(text):
    documents = []
    documents.append({
        'id': 0,
        'text': text
    })

    param = {'documents': documents}
    headers = {"Ocp-Apim-Subscription-Key": txt_analysis_key}
    response = requests.post(key_phrase_api_url, headers=headers, json=param)
    keyPhrasesJson = response.json()
    # pprint(keyPhrasesJson)
    numKeyPhrases = len(keyPhrasesJson['documents'][0]['keyPhrases'])
    # print(numKeyPhrases)
    return numKeyPhrases

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


def append_instance(question, answerable, imageData, blurMetric):
    question_pos_counts = pos_tag_text(question)
    # print("Description is: " + imageData['description']['captions'][0]['text'])
    desc_pos_counts = pos_tag_text(imageData['description']['captions'][0]['text'])
    
    feature_set['q_._count'].append(question_pos_counts['.'])
    feature_set['q_ADJ_count'].append(question_pos_counts['ADJ'])
    feature_set['q_ADP_count'].append(question_pos_counts['ADP'])
    feature_set['q_ADV_count'].append(question_pos_counts['ADV'])
    feature_set['q_CC_count'].append(question_pos_counts['CC'])
    feature_set['q_CONJ_count'].append(question_pos_counts['CONJ'])
    feature_set['q_DET_count'].append(question_pos_counts['DET'])
    feature_set['q_IN_count'].append(question_pos_counts['IN'])
    feature_set['q_JJ_count'].append(question_pos_counts['JJ'])
    feature_set['q_NN_count'].append(question_pos_counts['NN'])
    feature_set['q_NOUN_count'].append(question_pos_counts['NOUN'])
    feature_set['q_NUM_count'].append(question_pos_counts['NUM'])
    feature_set['q_PRON_count'].append(question_pos_counts['PRON'])
    feature_set['q_PRT_count'].append(question_pos_counts['PRT'])
    feature_set['q_RB_count'].append(question_pos_counts['RB'])
    feature_set['q_VERB_count'].append(question_pos_counts['VERB'])
    feature_set['q_X_count'].append(question_pos_counts['X'])

    feature_set['desc_._count'].append(desc_pos_counts['.'])
    feature_set['desc_ADJ_count'].append(desc_pos_counts['ADJ'])
    feature_set['desc_ADP_count'].append(desc_pos_counts['ADP'])
    feature_set['desc_ADV_count'].append(desc_pos_counts['ADV'])
    feature_set['desc_CC_count'].append(desc_pos_counts['CC'])
    feature_set['desc_CONJ_count'].append(desc_pos_counts['CONJ'])
    feature_set['desc_DET_count'].append(desc_pos_counts['DET'])
    feature_set['desc_IN_count'].append(desc_pos_counts['IN'])
    feature_set['desc_JJ_count'].append(desc_pos_counts['JJ'])
    feature_set['desc_NN_count'].append(desc_pos_counts['NN'])
    feature_set['desc_NOUN_count'].append(desc_pos_counts['NOUN'])
    feature_set['desc_NUM_count'].append(desc_pos_counts['NUM'])
    feature_set['desc_PRON_count'].append(desc_pos_counts['PRON'])
    feature_set['desc_PRT_count'].append(desc_pos_counts['PRT'])
    feature_set['desc_RB_count'].append(desc_pos_counts['RB'])
    feature_set['desc_VERB_count'].append(desc_pos_counts['VERB'])
    feature_set['desc_X_count'].append(desc_pos_counts['X'])

    feature_set['is_image_BnW'].append(int(imageData["color"]["isBwImg"]))
    feature_set['image_bg_dom_color_BLACK'].append(0)
    feature_set['image_bg_dom_color_BLUE'].append(0)
    feature_set['image_bg_dom_color_BROWN'].append(0)
    feature_set['image_bg_dom_color_GREY'].append(0)
    feature_set['image_bg_dom_color_GREEN'].append(0)
    feature_set['image_bg_dom_color_ORANGE'].append(0)
    feature_set['image_bg_dom_color_PINK'].append(0)
    feature_set['image_bg_dom_color_PURPLE'].append(0)
    feature_set['image_bg_dom_color_RED'].append(0)
    feature_set['image_bg_dom_color_TEAL'].append(0)
    feature_set['image_bg_dom_color_WHITE'].append(0)
    feature_set['image_bg_dom_color_YELLOW'].append(0)
    feature_set['image_fg_dom_color_BLACK'].append(0)
    feature_set['image_fg_dom_color_BLUE'].append(0)
    feature_set['image_fg_dom_color_BROWN'].append(0)
    feature_set['image_fg_dom_color_GREY'].append(0)
    feature_set['image_fg_dom_color_GREEN'].append(0)
    feature_set['image_fg_dom_color_ORANGE'].append(0)
    feature_set['image_fg_dom_color_PINK'].append(0)
    feature_set['image_fg_dom_color_PURPLE'].append(0)
    feature_set['image_fg_dom_color_RED'].append(0)
    feature_set['image_fg_dom_color_TEAL'].append(0)
    feature_set['image_fg_dom_color_WHITE'].append(0)
    feature_set['image_fg_dom_color_YELLOW'].append(0)


    if (imageData["color"]["dominantColorBackground"] == "Black"):
        feature_set['image_bg_dom_color_BLACK'] = feature_set['image_bg_dom_color_BLACK'][:-1] + [1]
    elif (imageData["color"]["dominantColorBackground"] == "Blue"):
        feature_set['image_bg_dom_color_BLUE'] = feature_set['image_bg_dom_color_BLUE'][:-1] + [1]
    elif (imageData["color"]["dominantColorBackground"] == "Brown"):
        feature_set['image_bg_dom_color_BROWN'] = feature_set['image_bg_dom_color_BROWN'][:-1] + [1]
    elif (imageData["color"]["dominantColorBackground"] == "Grey"):
        feature_set['image_bg_dom_color_GREY'] = feature_set['image_bg_dom_color_GREY'][:-1] + [1]
    elif (imageData["color"]["dominantColorBackground"] == "Green"):
        feature_set['image_bg_dom_color_GREEN'] = feature_set['image_bg_dom_color_GREEN'][:-1] + [1]
    elif (imageData["color"]["dominantColorBackground"] == "Orange"):
        feature_set['image_bg_dom_color_ORANGE'] = feature_set['image_bg_dom_color_ORANGE'][:-1] + [1]
    elif (imageData["color"]["dominantColorBackground"] == "Pink"):
        feature_set['image_bg_dom_color_PINK'] = feature_set['image_bg_dom_color_PINK'][:-1] + [1]
    elif (imageData["color"]["dominantColorBackground"] == "Purple"):
        feature_set['image_bg_dom_color_PURPLE'] = feature_set['image_bg_dom_color_PURPLE'][:-1] + [1]
    elif (imageData["color"]["dominantColorBackground"] == "Red"):
        feature_set['image_bg_dom_color_RED'] = feature_set['image_bg_dom_color_RED'][:-1] + [1]
    elif (imageData["color"]["dominantColorBackground"] == "Teal"):
        feature_set['image_bg_dom_color_TEAL'] = feature_set['image_bg_dom_color_TEAL'][:-1] + [1]
    elif (imageData["color"]["dominantColorBackground"] == "White"):
        feature_set['image_bg_dom_color_WHITE'] = feature_set['image_bg_dom_color_WHITE'][:-1] + [1]
    elif (imageData["color"]["dominantColorBackground"] == "Yellow"):
        feature_set['image_bg_dom_color_YELLOW'] = feature_set['image_bg_dom_color_YELLOW'][:-1] + [1]
    else:
        print("UNKNOWN COLOR BG")

    if (imageData["color"]["dominantColorForeground"] == "Black"):
        feature_set['image_fg_dom_color_BLACK'] = feature_set['image_fg_dom_color_BLACK'][:-1] + [1]
    elif (imageData["color"]["dominantColorForeground"] == "Blue"):
        feature_set['image_fg_dom_color_BLUE'] = feature_set['image_fg_dom_color_BLUE'][:-1] + [1]
    elif (imageData["color"]["dominantColorForeground"] == "Brown"):
        feature_set['image_fg_dom_color_BROWN'] = feature_set['image_fg_dom_color_BROWN'][:-1] + [1]
    elif (imageData["color"]["dominantColorForeground"] == "Grey"):
        feature_set['image_fg_dom_color_GREY'] = feature_set['image_fg_dom_color_GREY'][:-1] + [1]
    elif (imageData["color"]["dominantColorForeground"] == "Green"):
        feature_set['image_fg_dom_color_GREEN'] = feature_set['image_fg_dom_color_GREEN'][:-1] + [1]
    elif (imageData["color"]["dominantColorForeground"] == "Orange"):
        feature_set['image_fg_dom_color_ORANGE'] = feature_set['image_fg_dom_color_ORANGE'][:-1] + [1]
    elif (imageData["color"]["dominantColorForeground"] == "Pink"):
        feature_set['image_fg_dom_color_PINK'] = feature_set['image_fg_dom_color_PINK'][:-1] + [1]
    elif (imageData["color"]["dominantColorForeground"] == "Purple"):
        feature_set['image_fg_dom_color_PURPLE'] = feature_set['image_fg_dom_color_PURPLE'][:-1] + [1]
    elif (imageData["color"]["dominantColorForeground"] == "Red"):
        feature_set['image_fg_dom_color_RED'] = feature_set['image_fg_dom_color_RED'][:-1] + [1]
    elif (imageData["color"]["dominantColorForeground"] == "Teal"):
        feature_set['image_fg_dom_color_TEAL'] = feature_set['image_fg_dom_color_TEAL'][:-1] + [1]
    elif (imageData["color"]["dominantColorForeground"] == "White"):
        feature_set['image_fg_dom_color_WHITE'] = feature_set['image_fg_dom_color_WHITE'][:-1] + [1]
    elif (imageData["color"]["dominantColorForeground"] == "Yellow"):
        feature_set['image_fg_dom_color_YELLOW'] = feature_set['image_fg_dom_color_YELLOW'][:-1] + [1]
    else:
        print("UNKNOWN COLOR FG")    

    feature_set['q_string_color_BLACK'].append(0)
    feature_set['q_string_color_BLUE'].append(0)
    feature_set['q_string_color_BROWN'].append(0)
    feature_set['q_string_color_GREY'].append(0)
    feature_set['q_string_color_GREEN'].append(0)
    feature_set['q_string_color_ORANGE'].append(0)
    feature_set['q_string_color_PINK'].append(0)
    feature_set['q_string_color_PURPLE'].append(0)
    feature_set['q_string_color_RED'].append(0)
    feature_set['q_string_color_TEAL'].append(0)
    feature_set['q_string_color_WHITE'].append(0)
    feature_set['q_string_color_YELLOW'].append(0)

    if ("BLACK" in question.upper()):
        feature_set['q_string_color_BLACK'] = feature_set['q_string_color_BLACK'][:-1] + [1]
    if ("BLUE" in question.upper()):
        feature_set['q_string_color_BLUE'] = feature_set['q_string_color_BLUE'][:-1] + [1]
    if ("BROWN" in question.upper()):
        feature_set['q_string_color_BROWN'] = feature_set['q_string_color_BROWN'][:-1] + [1]
    if ("GREY" in question.upper()):
        feature_set['q_string_color_GREY'] = feature_set['q_string_color_GREY'][:-1] + [1]
    if ("GREEN" in question.upper()):
        feature_set['q_string_color_GREEN'] = feature_set['q_string_color_GREEN'][:-1] + [1]
    if ("ORANGE" in question.upper()):
        feature_set['q_string_color_ORANGE'] = feature_set['q_string_color_ORANGE'][:-1] + [1]
    if ("PINK" in question.upper()):
        feature_set['q_string_color_PINK'] = feature_set['q_string_color_PINK'][:-1] + [1]
    if ("PURPLE" in question.upper()):
        feature_set['q_string_color_PURPLE'] = feature_set['q_string_color_PURPLE'][:-1] + [1]
    if ("RED" in question.upper()):
        feature_set['q_string_color_RED'] = feature_set['q_string_color_RED'][:-1] + [1]
    if ("TEAL" in question.upper()):
        feature_set['q_string_color_TEAL'] = feature_set['q_string_color_TEAL'][:-1] + [1]
    if ("WHITE" in question.upper()):
        feature_set['q_string_color_WHITE'] = feature_set['q_string_color_WHITE'][:-1] + [1]
    if ("YELLOW" in question.upper()):
        feature_set['q_string_color_YELLOW'] = feature_set['q_string_color_YELLOW'][:-1] + [1]

    feature_set['image_has_faces'].append(int(len(imageData["faces"])))
    feature_set['image_num_faces'].append(len(imageData["faces"]))

    feature_set['image_has_text'].append(0)
    for item in imageData['categories']:
        if item['name'] == "text_":
            feature_set['image_has_text'] = feature_set['image_has_text'][:-1] + [1]

    feature_set['blurriness'].append(blurMetric)

    feature_set['q_sentiment'].append(get_sentiments(question))
    feature_set['q_num_key_phrases'].append(get_num_key_phrases(question))
    feature_set['desc_num_key_phrases'].append(get_num_key_phrases(imageData['description']['captions'][0]['text']))
    # 'q_num_key_phrases' : [],
    # 'desc_num_key_phrases' : []

    target_set['target'].append(answerable)

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
feature_set = {
    'q_._count' : [],
    'q_ADJ_count': [],
    'q_ADP_count': [],
    'q_ADV_count': [],
    'q_CC_count': [],
    'q_CONJ_count': [],
    'q_DET_count': [],
    'q_IN_count': [],
    'q_JJ_count': [],
    'q_NN_count': [],
    'q_NOUN_count': [],
    'q_NUM_count': [],
    'q_PRON_count': [],
    'q_PRT_count': [],
    'q_RB_count': [],
    'q_VERB_count': [],
    'q_X_count': [],

    'desc_._count' : [],
    'desc_ADJ_count': [],
    'desc_ADP_count': [],
    'desc_ADV_count': [],
    'desc_CC_count': [],
    'desc_CONJ_count': [],
    'desc_DET_count': [],
    'desc_IN_count': [],
    'desc_JJ_count': [],
    'desc_NN_count': [],
    'desc_NOUN_count': [],
    'desc_NUM_count': [],
    'desc_PRON_count': [],
    'desc_PRT_count': [],
    'desc_RB_count': [],
    'desc_VERB_count': [],
    'desc_X_count': [],

    'is_image_BnW' : [],
    'image_bg_dom_color_BLACK' : [],
    'image_bg_dom_color_BLUE' : [],
    'image_bg_dom_color_BROWN' : [],
    'image_bg_dom_color_GREY' : [],
    'image_bg_dom_color_GREEN' : [],
    'image_bg_dom_color_ORANGE' : [],
    'image_bg_dom_color_PINK' : [],
    'image_bg_dom_color_PURPLE' : [],
    'image_bg_dom_color_RED' : [],
    'image_bg_dom_color_TEAL' : [],
    'image_bg_dom_color_WHITE' : [],
    'image_bg_dom_color_YELLOW' : [],

    'image_fg_dom_color_BLACK' : [],
    'image_fg_dom_color_BLUE' : [],
    'image_fg_dom_color_BROWN' : [],
    'image_fg_dom_color_GREY' : [],
    'image_fg_dom_color_GREEN' : [],
    'image_fg_dom_color_ORANGE' : [],
    'image_fg_dom_color_PINK' : [],
    'image_fg_dom_color_PURPLE' : [],
    'image_fg_dom_color_RED' : [],
    'image_fg_dom_color_TEAL' : [],
    'image_fg_dom_color_WHITE' : [],
    'image_fg_dom_color_YELLOW' : [],     

    'q_string_color_BLACK' : [],
    'q_string_color_BLUE' : [],
    'q_string_color_BROWN' : [],
    'q_string_color_GREY' : [],
    'q_string_color_GREEN' : [],
    'q_string_color_ORANGE' : [],
    'q_string_color_PINK' : [],
    'q_string_color_PURPLE' : [],
    'q_string_color_RED' : [],
    'q_string_color_TEAL' : [],
    'q_string_color_WHITE' : [],
    'q_string_color_YELLOW' : [],

    'image_has_faces' : [],
    'image_num_faces' : [],

    'image_has_text' : [],
    'blurriness' : [],

    'q_sentiment' : [],
    'q_num_key_phrases' : [],
    'desc_num_key_phrases' : []
}

target_set = {
    'target' : []
}

# nltk.download('tagsets')
# text = 'And now for something completely different'
# pos_counts = pos_tag_text(text)
# pprint(pos_counts)

def main():
    print("Starting main")
    trainingData = json.load(open('data/train.json'))
    # pprint(trainingData[2])
    
    for ind in range(0,300):
        image_url = "https://cvc.ischool.utexas.edu/~dannag/VizWiz/Images/VizWiz_train_" + str(ind).zfill(12) + ".jpg"
        # image_url = "https://cvc.ischool.utexas.edu/~dannag/VizWiz/Images/VizWiz_train_000000000005.jpg"
        print("Starting " + image_url + "...")
        question = trainingData[ind]['question']
        answer = trainingData[ind]['answerable']
        # print("Q: " + question)
        # print("Answerable? " + str(answer))

        image = skimage.io.imread(image_url)
        imgAnalysis = analyze_image(image_url)
        # pprint(imgAnalysis)

        # print("\n --- FEATURES --- \n")

        # features = extract_features(imgAnalysis)
        # pprint(features)

        blur = detect_blur(image)
        # print("\n\nBlurriness index: " + str(blur))

        append_instance(question, answer, imgAnalysis, blur)
        # pprint(feature_set)

    # print("\n --- DATA FRAME --- \n")

    features_df = DataFrame(data=feature_set)
    target_df = DataFrame(data=target_set)
    # display(df)
    features_df.to_csv("feature_set.csv")
    target_df.to_csv("target.csv")    

if __name__ == '__main__':
    main()
    # get_sentiments("This is a fantastic question, but what color is that shirt?")
    # get_num_key_phrases("My very eager mother was keen to eat pizza with me and my friend.")