import cv2 as cv
import numpy as np
import zipfile

path = "YOUR ZIP ARCHIVE"
needle_img = cv.imread("OBJECT YOU NEED TO DETECT", cv.IMREAD_UNCHANGED)
treshold = 0.8 # You can change it in (0: 1) for better results


with zipfile.ZipFile(path) as archive:
    unzipped_lst = archive.infolist()
    for i in range(len(unzipped_lst)):
        ifile = archive.open(unzipped_lst[i])
        img_bytes = ifile.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        haystack_img = cv.imdecode(nparr, cv.IMREAD_UNCHANGED)
        result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if max_val > treshold:
            needle_w = needle_img.shape[1]
            needle_h = needle_img.shape[0]
            print("Found needle")
            top_left = max_loc
            bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)
            cv.rectangle(haystack_img, top_left, bottom_right, color=(0, 255, 0), thickness=2, lineType=cv.LINE_4)
            cv.imwrite(f"{archive.namelist()[i]} with object.jpg", haystack_img)
        else:
            print("Needle not found")