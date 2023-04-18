#Metrics to compare image quality
import numpy as np
from skimage.metrics import structural_similarity as ssim_

def ssim(A, B): #Wrapper for skimage ssim func with default parameters that I use
    return ssim_(
        A,
        B,
        data_range=255,
        gaussian_weights=True,
        win_size=11,
        K1=0.01,
        K2=0.03
    )

class ConfMat:
    def __init__(self, A, B, thresh = 128):
        ha, wa = A.shape
        hb, wb = B.shape
        assert ha == hb
        assert wa == wb
        A_thresh = A > thresh
        B_thresh = B > thresh
        tp, fp, tn, fn = 0, 0, 0, 0
        for y in range(ha):
            for x in range(wa):
                if A_thresh[y, x]:
                    if B_thresh[y, x]:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if B_thresh[y, x]:
                        fn += 1
                    else:
                        tn += 1
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn
    
    def precision(self):
        return self.tp/(self.tp+self.fp)

    def recall(self):
        return self.tp/(self.tp+self.fn)

    def fpr(self):
        return self.fp/(self.fp+self.tn)

    def tnr(self):
        return self.tn/(self.tn+self.fp)

    def f1_score(self):
        return 2*self.tp/(2*self.tp+self.fn+self.fp)

    def acc(self):
        return (self.tp+self.tn)/(self.tp+self.tn+self.fp+self.fn)