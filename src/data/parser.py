# parses raw dataset files into image matrices and labels
# handles datasets (train/validation/test)

import os # we will build our paths for text acces
import numpy as np # we will be using numpy for our feature arrays

#here are our image  dimentions on our files we are parsing, double check to make the dimesnions are right
DIGIT_DATA_HEIGHT = 28
DIGIT_DATA_WIDTH = 28

FACE_DATA_HEIGHT = 70
FACE_DATA_WIDTH = 60

ROOT_DIR = os.path.abspath( os.path.join(os.path.dirname(__file__), '..', '..',)) #each dot goes  deeper into the file system just enough to locate where the data is, subject to changhe depeding on how the file system works
print(f"DEBUG: Looking for data in: {ROOT_DIR}")
# theses are the file path to each respecitve files starts goes to dataset name we set -> then the name of folder -> the specific file

DATA_PATHS = {
    'digit': {
        'train': ( os.path.join(ROOT_DIR, 'data', 'digitdata', 'trainingimages'), os.path.join(ROOT_DIR, 'data', 'digitdata', 'traininglabels')),
        'vali': (os.path.join(ROOT_DIR, 'data', 'digitdata', 'validationimages'), os.path.join(ROOT_DIR, 'data', 'digitdata', 'validationlabels')),
        'test': (os.path.join(ROOT_DIR, 'data', 'digitdata', 'testimages'), os.path.join(ROOT_DIR, 'data', 'digitdata', 'testlabels')),
    },
    'face': {
        'train': (os.path.join(ROOT_DIR, 'data', 'facedata', 'facedatatrain'), os.path.join(ROOT_DIR, 'data', 'facedata', 'facedatatrainlabels')),
        'vali': (os.path.join(ROOT_DIR, 'data', 'facedata', 'facedatavalidation'), os.path.join(ROOT_DIR, 'data', 'facedata', 'facedatavalidationlabels')),
        'test': (os.path.join(ROOT_DIR, 'data', 'facedata', 'facedatatest'), os.path.join(ROOT_DIR, 'data', 'facedata', 'facedatatestlabels')),
    },
}

#here we will have our version of parsing through the image files
def _parsing_images(image_filepath, image_height):#image file path will bne a string to get the path of the image text file loactions
                                                  # image height is a intergert for the number of lines that creates one image

    with open(image_filepath, 'r') as f:#open the file and read every line into a list the 'r' just sets the mode to read only 
        all_lines = f.readlines()

    all_lines = [line.rstrip('\n') for line in all_lines]#this will help clean up the way that the characters are organize, the rstrip('\n) only remove the right side  of the lines

    images = [] # this will hold all the parse images

# going through all lines starting of the image height and increments froim there
    for i in range(0, len(all_lines), image_height):
        chunk = all_lines[i : i + image_height] # this chunk represens the exact height of the image at the starting position all the way to the end of it, one complete image worth of text


        if len(chunk) < image_height:# if the chunk we grab is less than the image height we hit the end of the file or maybe running into an incomplete image
            continue# we skip it by continueing so we wont store a broken image

        if image_height == DIGIT_DATA_HEIGHT: #this if statement helps with understanding the width of the image of text to parse either a digit image or a face image width set.
            expected_width = DIGIT_DATA_WIDTH
        else:
            expected_width = FACE_DATA_WIDTH

        chunk = [row.ljust(expected_width) for row in chunk] #sets each row to the new expected width

        image_grid = [list(row) for row in chunk] # this will convert each row into inficiaula characters the list(row) will be doing the spliting

        images.append(image_grid) # this will update the image frupd to now be a 2d list of characters

    return images #returning a list of 2d chars files one per image


#this will be the parsing of label specific files
def _parsing_labels(label_filepath):#label firls path is a string to the path of the lavel text file
    with open(label_filepath, 'r') as f: #opening the files and setting as read only 
        lines = f.readlines() 

    labels = []
    for line in lines:
        stripped = line.strip() # remove our spaces on both sides

        if stripped == '':
            continue #skipping empty lines

        labels.append(int(stripped)) # we convert opur string character of the numbers to actual integers

    return labels # returing a list of 2d integers one per label file


# these will be our functions that teammates use to call so that they can import the data much simpler and cleaner

def _load(dataset, split): # this helper function will be used by the load function of the digit and face data respectfully
                            # data set is the string to set the digit or face
                            # split is the string to set the train, val or test
    
    #this is an exceptions for validation the inputs of both of the pathways 
    assert dataset in DATA_PATHS,  f"Wrong dataset '{dataset}'. Must be 'digit' or 'face' ." 
    assert split in DATA_PATHS[dataset],  f"Wrong split '{split}'. Needs to be 'train'. 'vali', or 'test'."

    image_file, label_file = DATA_PATHS[dataset][split]#unpacking the files paths for the dataset and split combo

    height = DIGIT_DATA_HEIGHT if dataset == 'digit' else FACE_DATA_HEIGHT #choosing the right height based on the dataset

    #this will actaully run the parsing functions
    images = _parsing_images(image_file, height)
    labels = _parsing_labels(label_file)
# A sanity check for the numbers in images must be the same as the number of labels, if provoked  the height is wrong or the file is corrupted
    assert len(images) == len(labels), (f"[PARSER ERROR] {dataset}/{split}: " 
                                        f"Got {len(images)} images but {len(labels)} labels.\n" 
                                        f"   -> Check that {('DIGIT_DATA_HEIGHT' if dataset == 'digit' else 'FACE_DATA_HEIGHT')}" 
                                        f"is set correctly(currently {height})."
                                        )
    
    print(f"[parser] Loaded {len(images)} {dataset} images ({split} split)")
    return images, labels


def load_digit_data(split = 'train'):#this will load in the digit images and labels for the specific split a split is either the train,val or test folders
    return _load('digit', split)

def load_face_data(split = 'train'):#this will load in the face images and labels for the specific split like train, val or test
    return _load('face', split)



# ─────────────────────────────────────────────────────────────────────────────
# QUICK VISUAL VERIFICATION
# Run this file directly to confirm your parser works:
#   python src/data/parser.py
# this is just testing super irrelevant to what yall have to do maybe delete this before we turn this in
#depending how you use your algorithm some of these might be useful to access data 
# ─────────────────────────────────────────────────────────────────────────────

def _print_image(image_grid, label=None):
    """Prints one parsed image to the terminal for visual inspection."""
    if label is not None:
        print(f"  Label: {label}")
    for row in image_grid:
        print('  ' + ''.join(row))  # ''.join turns ['#',' ','+'] → "#  +"
    print()


if __name__ == '__main__':

    print("=" * 50)
    print("DIGIT DATA TEST")
    print("=" * 50)

    digit_images, digit_labels = load_digit_data('train')

    # Print the first 3 digit images so you can visually verify them
    for i in range(3):
        print(f"Image #{i}:")
        _print_image(digit_images[i], digit_labels[i])

    print("=" * 50)
    print("FACE DATA TEST")
    print("=" * 50)

    face_images, face_labels = load_face_data('train')

    # Print just the first face image
    print(f"Image #0:")
    _print_image(face_images[0], face_labels[0])



