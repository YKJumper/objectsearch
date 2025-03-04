
import numpy as np
import math
import cv2 as cv
from matplotlib import pyplot as plt
scaleStep = 0.9
templ_shift = 0.0

def dist(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

def lst2Str(s):
     str1 = ""
     for n in s:
        str1 += str(n) + ","
     return str1

def matchScaledTemplate(template, img, meth, template_ver, zoom_step, min_degree):
    Wi = img.shape[1]
    Hi = img.shape[0]
    Wtv = template_ver.shape[1]
    Htv = template_ver.shape[0]
    maxScale = min(int(math.log(2*Wtv/Wi, zoom_step)), int(math.log(2*Htv/Hi, zoom_step)))
    scalePers = range(min_degree, maxScale)
    for j in scalePers:
        width = int(Wi * zoom_step ** j)
        height = int(Hi * zoom_step ** j)
        dim = (width, height)

        resized = cv.resize(img, dim, interpolation = cv.INTER_LINEAR)

        method = eval(meth)
        # Apply template Matching
        res = cv.matchTemplate(resized, template, method)
        res_ver = cv.matchTemplate(resized, template_ver, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        min_val_ver, max_val_ver, min_loc_ver, max_loc_ver = cv.minMaxLoc(res_ver)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
            top_left_ver = min_loc_ver
        else:
            top_left = max_loc
            top_left_ver = max_loc_ver
        dst = dist(top_left_ver, top_left)
        if dst <= h/8:
            bottom_right = (int((top_left[0] + w) / zoom_step ** j), int((top_left[1] + h) / zoom_step ** j))
            top_left = (int(top_left[0] / zoom_step ** j), int(top_left[1] / zoom_step ** j))
            return top_left, bottom_right, j, min_val, max_val, min_val_ver, max_val_ver
    return [-1, -1], [-1, -1], 0

# All the 6 methods for comparison in a list
# methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
meth = 'cv.TM_CCOEFF_NORMED'
# rectangle color in BGR 
color = (0, 0, 255) 
# Line thickness of 2 px 
thickness = 2
# images = ['', '', '', '', '', '', '', '','', '']
template = cv.imread('proxima2/target-t.jpg', cv.IMREAD_GRAYSCALE)
template_ver = cv.imread('proxima2/target-v.jpg', cv.IMREAD_GRAYSCALE)
assert template is not None, "file could not be read, check with os.path.exists()"
w = template.shape[1]
h = template.shape[0]

images_name = range(1,15)
degree = 0
for j in images_name:
    image_name = 'proxima2/Z'+str(j)+'.jpg'
    img = cv.imread(image_name, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    method = eval(meth)
    resPos = matchScaledTemplate(template, img, meth, template_ver, scaleStep, degree)
    top_left = resPos[0]
    bottom_right = resPos[1]


    if top_left[0] >= 0:
        # # Define the coordinates of the new template in the image
        scaleCoef = 1 / scaleStep**resPos[2]
        h_shift = 2 * h * scaleCoef
        w_shift = 2 * w * scaleCoef
        vx1, vy1, vx2, vy2 = int(resPos[0][0] - templ_shift * w_shift), int(resPos[0][1] - templ_shift * h_shift), int(resPos[0][0] + (1-templ_shift) * w_shift), int(resPos[0][1] + (1-templ_shift) * h_shift)
        # Extract the templates
        # сimg = cv.imread(image_name)
        template_ver = img[vy1:vy2, vx1:vx2].copy()
        template = template_ver[0:int(h * scaleCoef), 0:int(w * scaleCoef)].copy()
        w = template.shape[1]
        h = template.shape[0]

        # cv.imwrite('template-'+str(j)+'.jpg', template)
        cv.imwrite('template_ver-'+str(j)+'.jpg', template_ver)
        # template = cv.imread('template.jpg', cv.IMREAD_GRAYSCALE)
        # template_ver = cv.imread('template_ver.jpg', cv.IMREAD_GRAYSCALE)
     
        # # Show target template
        # print('Template dimensions : ',template.shape)
        # cv.imshow("Template image:", template)
        # cv.waitKey(0)
        
        # print('Template verification dimensions : ',template_ver.shape)
        # cv.imshow("Template verification image ", template_ver)
        # cv.waitKey(0)

        cv.rectangle(img, top_left, bottom_right, color, 2)
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img, cmap='gray')
        plt.title('Detected Point at '+image_name), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()
        cv.waitKey(0)
    print('Detected Point at '+image_name+':', resPos)