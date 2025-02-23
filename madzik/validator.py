import numpy as np
from madzik.loader import DataSetLoader
from madzik.processing import PLUMER


class Validator:
    def __init__(self, dataset_loader: DataSetLoader, model: PLUMER, params):
        self.dataset_loader = dataset_loader
        self.model = model
        self.params = params

    def evaluate(self):
        print(f"Evaluating {len(self.dataset_loader)} batches")
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        length = len(self.dataset_loader)
        for i in range(length):
            print(f"Batch {i+1}/{length}")
            x, y = self.dataset_loader[i]
            y_pred = self.model.model.predict(x)
            y_pred = np.round(y_pred)
            for val, pred in zip(y, y_pred):
                if val == 1 and pred == 1:
                    TP += 1
                elif val == 0 and pred == 0:
                    TN += 1
                elif val == 0 and pred == 1:
                    FP += 1
                elif val == 1 and pred == 0:
                    FN += 1
                print(
                    f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN} y: {val}, pred: {pred}")

        return {
            "sensitivity": self.sensitivity(TP, FN),
            "specificity": self.specificity(TN, FP),
            "precision": self.precision(TP, FP),
            "accuracy": self.accuracy(TP, TN, FP, FN),
            "fscore": self.fscore(TP, FP, FN),
            "NPV": self.NPV(TN, FN),
            "FPR": self.FPR(FP, TN),
            "IoU": self.IoU(TP, FP, FN),
            "MCC": self.MCC(TP, TN, FP, FN),
            "true_false_matrix": {"TP": TP, "TN": TN, "FP": FP, "FN": FN}
        }

    def sensitivity(self, TP, FN):
        return TP / (TP + FN)

    def specificity(self, TN, FP):
        return TN / (TN + FP)

    def precision(self, TP, FP):
        return TP / (TP + FP)

    def accuracy(self, TP, TN, FP, FN):
        return (TP + TN) / (TP + TN + FP + FN)

    def fscore(self, TP, FP, FN):
        return 2 * TP / (2 * TP + FP + FN)

    def NPV(self, TN, FN):
        return TN / (TN + FN)

    def FPR(self, FP, TN):
        return FP / (FP + TN)

    def IoU(self, TP, FP, FN):
        return TP / (TP + FP + FN)

    def MCC(self, TP, TN, FP, FN):
        return (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    def AUC(self, y_true, y_pred):
        pass

    def true_false_matrix(self, y_true, y_pred):
        return {
            "TP": np.sum(y_true * y_pred),
            "TN": np.sum((1 - y_true) * (1 - y_pred)),
            "FP": np.sum((1 - y_true) * y_pred),
            "FN": np.sum(y_true * (1 - y_pred))
        }
