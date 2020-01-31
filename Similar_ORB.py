from pathlib import Path
import cv2
import skimage
from skimage.io import imread
import numpy as np






def similar(original,image_to_compare):
    sift = cv2.ORB_create()
    kp_1, desc_1 = sift.detectAndCompute(original, None)
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(np.asarray(desc_1, np.float32), np.asarray(desc_2, np.float32), k=2)

    good_points = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good_points.append(m)

    # Define how similar they are
    number_keypoints = 0
    if len(kp_1) <= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)

    #print("Keypoints 1ST Image: " + str(len(kp_1)))
    #print("Keypoints 2ND Image: " + str(len(kp_2)))
    #print("GOOD Matches:", len(good_points))
    #print("How good it's the match: ", len(good_points) / number_keypoints * 100)
    #result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
    res = len(good_points) / number_keypoints * 100
    #cv2.imshow("result", cv2.resize(result, None, fx=0.4, fy=0.4))
    return res



container_path= "IC1/train/erreur/"
direc = Path(container_path)
orig="IC1/train/erreur/0374.jpg"
for file in direc.iterdir():
    original = skimage.io.imread(orig)
    image_to_compare= skimage.io.imread(file)
    print(file)
    cv2.imshow("T-1", cv2.resize(original, None, fx=0.4, fy=0.4))
    cv2.imshow("T", cv2.resize(image_to_compare, None, fx=0.4, fy=0.4))

    # 1) Check if 2 images are equals
    if original.shape == image_to_compare.shape:
        print("The images have same size and channels")
        difference = cv2.subtract(original, image_to_compare)
        b, g, r = cv2.split(difference)

        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            print("The images are completely Equal")
        else:

            diff = similar(original,image_to_compare)
            if diff<25:
                print("The images are NOT equal")
                orig = file
            else: print("similar equal to ", diff)
    else:
        diff = similar(original, image_to_compare)
        if diff <25:
            print("The images are NOT equal")
            orig = file
        else:
            print("similar equal to ", diff)
    cv2.waitKey(0)

cv2.destroyAllWindows()