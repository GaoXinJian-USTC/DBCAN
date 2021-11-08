import cv2
import DBCAN.datasets.pipelines
from DBCAN.apis import init_detector
from DBCAN.apis.inference import model_inference
from DBCAN.utils.model import revert_sync_batchnorm
import time


class DBCAN():

    def __init__(self,
                 recog_config='/home/will/Downloads/DBCAN/configs/DBCAN.py',
                 recog_ckpt='/home/will/Downloads/DBCAN_Models/best_5.pth',
                 device='cuda:0',
                 ):
        self.device = device
        self.recog_model = None
        self.recog_model = init_detector(
            recog_config, recog_ckpt, device=self.device)
        self.recog_model = revert_sync_batchnorm(self.recog_model)

    def inference(self, img):
        recog_result = model_inference(self.recog_model, img)  
        return recog_result

if __name__ == '__main__':
 
    model = DBCAN()
    img = cv2.imread("/media/will/6f586f18-792a-40fd-ada6-59702fb5dabc/gaoxinjian/dataset/ocr_benchmark/CUTE80/IMG/2.jpg")
    t1 = time.time()
    res = model.inference(img)
    t2 = time.time()
    print(res,f"\nTotal time usage {t2-t1} seconds.")