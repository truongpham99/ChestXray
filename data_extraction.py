""" Turns image data into embedding vectors. Returning a pickle file.
"""
from keras.applications import resnet50
import utils
import argparse 
import sys
import pickle

def main(args):
    if args.model == "resnet50":
        model = resnet50.ResNet50(include_top=False, input_shape=(224,224,3))
        image_size = 224
    
    dataset = utils.get_data(args.data_path)
    image_paths, labels = utils.get_image_paths_and_labels(dataset)
    images = utils.load_images(image_paths, image_size)
    
    last_output = model.predict(images)
    with open(args.model + "_" + args.set + ".pkl", "wb") as outfile:
        pickle.dump((last_output, labels), outfile)
        
    return

def argument_parser(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("data_path", type=str,
               help="Put in the path for the data")
    parser.add_argument("model", type=str, choices=["resnet50"],
               help="Type of the model to extract output")
    parser.add_argument("--set", type=str,
               help="Type of dataset (train, test, val)", default="")
    
    return parser.parse_args(argv)

if __name__ == "__main__":
    main(argument_parser(sys.argv[1:]))
