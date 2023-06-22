import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

if __name__ == '__main__':
    seg_path = os.path.join('2d', 'segmentations')
    pred_path = 'runs'

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    all_y = np.array([], dtype=np.uint8)
    all_fx = np.array([], dtype=np.uint8)

    for i in range(40, 51):
        y = cv2.imread(os.path.join(seg_path, f'{i}.png'))
        fx = cv2.imread(os.path.join(pred_path, f'{i}.png'))

        all_y = np.concatenate([all_y, np.asarray(y.flatten() / 255, dtype=np.uint8)], axis=0)
        all_fx = np.concatenate([all_fx, np.asarray(fx.flatten() / 255, dtype=np.uint8)], axis=0)

        TP += np.sum((y == 255) & (fx == 255))
        FP += np.sum((y == 0) & (fx == 255))
        TN += np.sum((y == 0) & (fx == 0))
        FN += np.sum((y == 255) & (fx == 0))

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    print(f"Accuracy: {accuracy}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")

    fpr, tpr, thres = roc_curve(all_y, all_fx)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig('ROC.png')
