from paddleocr import PaddleOCR
import cv2
from PIL import Image

ocr = PaddleOCR()
img_path = "use_imgs/312412.jpg"
result = ocr.ocr(img_path)



it = iter(result)
for i in range(len(result)):
    line = next(it)
    tr = line[0][0][1]
    bl = line[0][3][1]
    tl = line[0][0][0]
    br = line[0][1][0]
    img = Image.open(img_path)
    box = (tl, tr, br, bl)
    region = img.crop(box) # (left, upper, right, lower)
    region.save('./crop_imgs/crop'+str(i)+'.jpg')
    # region.show()
    # img = cv2.imread(img_path)
    # crop = img[int(tr):int(bl), int(tl):int(br)]
    # cv2.imwrite("IMG.img", crop)
