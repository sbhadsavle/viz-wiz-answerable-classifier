## code to generate a feature representation for VizWiz questions

import json
from pprint import pprint

def main():
	print("Starting main")
	trainingData = json.load(open('data/train.json'))
	pprint(trainingData[2])


if __name__ == '__main__':
    main()