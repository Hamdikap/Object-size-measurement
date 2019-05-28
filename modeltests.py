import cv2
from fastai.vision import *
import os

learn = load_learner(Path(''), Path('abc.pkl'))
dogruluk = 0

model_name = "teknorot grup "
f= open("E:/{}.txt".format(model_name),"w+")
for i in (os.listdir("C:/Users/Hamdi/Desktop/size-of-objects/grup/")):
    dosya, uzantı = os.path.splitext('{}'.format(i))
    img1 = cv2.imread("C:/Users/Hamdi/Desktop/size-of-objects/grup/{}".format(dosya + uzantı))
    if ("jpg" in uzantı) == True:
        pred_class, pred_idx, outputs = learn.predict(Image(pil2tensor(img1, np.float32).div_(255)))
        if dosya[0]== str(pred_class)[0]:
            dogruluk+=1


        f.write(
            "pred_class: {}\n".format(pred_class) +
            "outputs :  {}\n".format(outputs) +
            "image name : {}\n".format(dosya+uzantı) +
            "******************\n\n")
f.write("dogruluk : {}\n".format(dogruluk))

f.close()