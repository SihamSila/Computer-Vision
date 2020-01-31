from skimage.measure import compare_ssim
import skimage
from skimage.io import imread
import cv2
from pathlib import Path



# load the two input images
image_orig = cv2.imread("IC1/train/erreur/0374.jpg")
resized_orig = cv2.resize(image_orig, (300, 200))
gray_orig = cv2.cvtColor(resized_orig, cv2.COLOR_BGR2GRAY)
container_path= "IC1/train/erreur/"
direc = Path(container_path)


for file in direc.iterdir():
    image_mod = skimage.io.imread(file)
    resized_mod = cv2.resize(image_mod, (300, 200))
# convert the images to grayscale
    gray_mod = cv2.cvtColor(resized_mod, cv2.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(gray_orig, gray_mod, full=True)
    diff = (diff * 255).astype("uint8")
    #print("Structural Similarity Index: {}".format(score))
    if score < 0.55:
        print("Motion détection", score)
        image_orig = skimage.io.imread(file)
        resized_orig = cv2.resize(image_orig, (300, 200))
        gray_orig = cv2.cvtColor(resized_orig, cv2.COLOR_BGR2GRAY)
    else : print("Images similaires")