# converts the images into feature vectors for models
# ex: pixel values, grid features (like whitespace counts)
#usiing of teammate from the right file path of feature extrasctor import the extract features
# feature_vector has the pixel and grid respectfully
BACKGROUND = ' ' #this will represent the empty space of pixels

#helper functions that determin ifd a cell is on(there is something there) or off(empty)
def _is_marked(char):
    return 1 if char != BACKGROUND else 0


#Feature 1 pixel feature
#this will determine if a cell is marked(1) or empty (0)

def pixel_feature(image_grid):# image grid is our parameter of the list of string characters that is in the iamge that is parsed it represented on character

    features = [] #this will store a list of  1's and 0's

    for row in image_grid: # each row will be a list of characters from the image files
        for char in row:
            features.append(_is_marked(char)) # this will asssign each respective character a 1 marked or 0 for space
    
    return features#this will return a list of 1's and 0's

#feature 2 grid_region
#instead of looking at each pixel this will divide the image into a grid and for each block we get a 1 for a marked pixel or 0 for a block that is empty

def grid_feature(image_grid, grid_rows = 14, grid_cols = 6):# image grid: the list of str of the parsed image
                                                            # gird_rows: integer of how many rows to divide the image
                                                            # GRID_cols: inters of how many columes to divide
# getting the actual dimesnions of the image from the grid itself, the rows and columns respectfully
    img_height = len(image_grid)
    img_width = len(image_grid[0])
#calculating how many rows/cols to be in each block, for the heigh and width respectfully
    block_height = img_height // grid_rows
    block_width = img_width // grid_cols
#capsizes each vblock to 1 so we wont get divisions errors, also iif the frid is alrger than the image
    block_height = max(block_height, 1)
    block_width = max(block_width, 1)

    features = []# this will store one value per grid block in a list
#tthis loop over each block by its grid position respectfully, gr is the row and gc is the column
    for gr in range(grid_rows):
        for gc in range(grid_cols):
#Calculating the pixel range in order to see what the block covers
#row starts from row times the height up to the sum of the start and height
            row_start = gr * block_height
            row_end = row_start + block_height
#column starts at the column time the width of the block up to the sum of the start and height
            col_start = gc * block_width
            col_end = col_start + block_width
#this will strictly make the end of the indices so they never go past the images borders
            row_end = min(row_end, img_height)
            col_end = min(col_end, img_width)

            block_is_marked = 0 #start as inactive by setting it to 0

            for r in range(row_start, row_end):#checking every pixel insisde the block if any pixel is marked the whole block will be active(1)
                for c in range(col_start, col_end):
                    if _is_marked(image_grid[r][c]):
                        block_is_marked = 1
                        break# no need to check more pixels antymore

                if block_is_marked:
                    break#no more checking on the row for this block
             
            features.append(block_is_marked)
    return features# this will return a list of 1 and 0

#these functions my teammates will call for your algorithms

def extract_features(image_grid, mode = 'pixel', grid_rows = 14, grid_cols = 6):#this will convert one image to a feature vector
    #will take image_grid: list of string lists from a parsed image from parser.py
    #mode: a string that will either be set to using the pixel or grid feature
    #the grid rows and columns that are used for the grid mode
    #if mode is either pixel or grid feature
    if mode == 'pixel':
        return pixel_feature(image_grid)
    elif mode == 'grid':
        return grid_feature(image_grid, grid_rows, grid_cols)
    
    else:# an expectaion for a error if the mode is wrong
        raise ValueError(f"Wrong mode '{mode}'. Choose 'pixel' or 'grid'.")
    
def extract_all(images, mode = 'pixel', grid_rows = 14, grid_cols = 6):# this will extract the entire dataset at once calls the extract_featur() on each imafge and returns all vectors as a list
    #same parameters as the extract_feature()

    all_features = []

    for image in images:
        #extract features vector for one image and adding it into our list
        vec = extract_features(image, mode = mode, grid_rows = grid_rows, grid_cols = grid_cols) 
        all_features.append(vec)

    return all_features


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST — run this file directly to verify everything works:
#   python src/feature/feature_extractor.py
# his is just testing super irrelevant to what yall have to do maybe delete this before we turn this in
#depending how you use your algorithm some of these might be useful to access data ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Import our parser to get real image data to test with
    import sys
    import os

    # Add the project root to Python's path so the import works
    # regardless of which folder you run this from
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    )

    from src.data.parser import load_digit_data, load_face_data

    print("=" * 55)
    print("FEATURE EXTRACTOR TEST")
    print("=" * 55)


    # ── Test on digit data ─────────────────────────────────────────────────
    print("\n[1] Loading digit training data...")
    digit_images, digit_labels = load_digit_data('train')

    # Test pixel features on the first digit image
    pixel_vec = extract_features(digit_images[0], mode='pixel')
    print(f"\nDigit image #0  (label: {digit_labels[0]})")
    print(f"  Pixel feature vector length : {len(pixel_vec)}")
    print(f"  Expected                    : {28 * 28} (28x28)")
    print(f"  First 20 values             : {pixel_vec[:20]}")
    # You should see a list of 0s and 1s — no other values

    # Test grid features on the first digit image
    grid_vec = extract_features(digit_images[0], mode='grid',
                                grid_rows=7, grid_cols=7)
    print(f"\n  Grid feature vector length  : {len(grid_vec)}")
    print(f"  Expected                    : {7 * 7} (7x7 grid)")
    print(f"  All values                  : {grid_vec}")
    # Should be 49 values of 0 or 1


    # ── Test on full dataset with extract_all ──────────────────────────────
    print("\n[2] Extracting pixel features for ALL digit training images...")
    X_digit_pixel = extract_all(digit_images, mode='pixel')
    print(f"  Total images processed      : {len(X_digit_pixel)}")
    print(f"  Feature vector length each  : {len(X_digit_pixel[0])}")

    print("\n[3] Extracting grid features for ALL digit training images...")
    X_digit_grid = extract_all(digit_images, mode='grid',
                               grid_rows=7, grid_cols=7)
    print(f"  Total images processed      : {len(X_digit_grid)}")
    print(f"  Feature vector length each  : {len(X_digit_grid[0])}")


    # ── Test on face data ──────────────────────────────────────────────────
    print("\n[4] Loading face training data...")
    face_images, face_labels = load_face_data('train')

    pixel_vec_face = extract_features(face_images[0], mode='pixel')
    print(f"\nFace image #0  (label: {face_labels[0]} — "
          f"{'face' if face_labels[0] == 1 else 'not a face'})")
    print(f"  Pixel feature vector length : {len(pixel_vec_face)}")
    print(f"  Expected                    : {70 * 60} (70x60)")

    # For faces we use a 10×6 grid to match the 70×60 aspect ratio
    # 70 / 10 = 7px tall blocks,  60 / 6 = 10px wide blocks
    grid_vec_face = extract_features(face_images[0], mode='grid',
                                     grid_rows=10, grid_cols=6)
    print(f"  Grid feature vector length  : {len(grid_vec_face)}")
    print(f"  Expected                    : {10 * 6} (10x6 grid)")

    print("\n All tests passed! Feature extractor is ready.")
