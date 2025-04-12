import torch

class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, true, pred):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=true.device)

        with torch.inference_mode():
            # T, P
            # 0, 0 => 2*0 + 0 = 0 True Negative
            # 0, 1 => 2*0 + 1 = 1 False Positive
            # 1, 0 => 2*1 + 0 = 2 False Negative
            # 1, 1 => 2*1 + 1 = 3 True Positive
            k = (true >= 0) & (true < n)
            inds = n * true[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def compute(self):
        # Confusion Matrix
        # [[TN, FP],
        #  [FN, TP]]
        h = self.mat.float()

        # TN, TP
        # TN -> Correctly predicted as class 0
        # TP -> Correctly predicted as class 1
        diag = torch.diag(h)

        # Overall accuracy
        # (TN + TP) / (TN + FP + FN + TP)
        acc_global = diag.sum() / h.sum()

        # Actual Negative, Actual Positive
        # (TN + FP) -> Actual Negative -> # of pixels that are class 0
        # (FN + TP) -> Actual Positive -> # of pixels that are class 1
        hsum1 = h.sum(1)

        # Predicted Negative, Predicted Positive
        # (TN + FN) -> Predicted Negative -> # of pixels predicted as class 0
        # (FP + TP) -> Predicted Positive -> # of pixels predicted as class 1
        hsum0 = h.sum(0)

        # Accuracy of class 0, Accuracy of class 1
        # (TN) / (TN + FP) -> Accuracy of class 0 
        # (TN + FP = Actual Negative -> # of pixels that are class 0)
        #
        # (TP) / (FN + TP) -> Accuracy of class 1  
        # (FN + TP = Actual Positive -> # of pixels that are class 1)
        acc = diag / hsum1

        # IoU for class 0, IoU for class 1
        # (TN) / ((TN + FP) + (TN + FN) - TN) => TN / (TN + FP + FN) -> IoU for class 0
        # (TP) / ((FN + TP) + (FP + TP) - TP) => TP / (FN + FP + TP) -> IoU for class 1
        iou = diag / (hsum1 + hsum0 - diag)
        return acc_global, acc, iou