import glob
import tensorflow as tf
import numpy as np

from sklearn.metrics import confusion_matrix

def readData(datasets_dir):
    img_paths = glob.glob(datasets_dir)
    label = []
    for i in range(len(img_paths)):
        if 'covid' in img_paths[i]:
            label.append([0, 1])
        else:
            label.append([1, 0])
    
    return img_paths, label

def lsgan_loss():
    mse = tf.losses.MeanSquaredError()

    def d_loss_fn(r_logit, f_logit):
        r_loss = mse(tf.ones_like(r_logit), r_logit)
        f_loss = mse(tf.zeros_like(f_logit), f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = mse(tf.ones_like(f_logit), f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn

def immerge(images, n_rows=None, n_cols=None, padding=0, pad_value=0):
    """Merge images to an image with (n_rows * h) * (n_cols * w).

    Parameters
    ----------
    images : numpy.array or object which can be converted to numpy.array
        Images in shape of N * H * W(* C=1 or 3).

    """
    images = np.array(images)
    n = images.shape[0]
    if n_rows:
        n_rows = max(min(n_rows, n), 1)
        n_cols = int(n - 0.5) // n_rows + 1
    elif n_cols:
        n_cols = max(min(n_cols, n), 1)
        n_rows = int(n - 0.5) // n_cols + 1
    else:
        n_rows = int(n ** 0.5)
        n_cols = int(n - 0.5) // n_rows + 1

    h, w = images.shape[1], images.shape[2]
    shape = (h * n_rows + padding * (n_rows - 1),
             w * n_cols + padding * (n_cols - 1))
    if images.ndim == 4:
        shape += (images.shape[3],)
    img = np.full(shape, pad_value, dtype=images.dtype)

    for idx, image in enumerate(images):
        i = idx % n_cols
        j = idx // n_cols
        img[j * (h + padding):j * (h + padding) + h,
            i * (w + padding):i * (w + padding) + w, ...] = image

    return img

def computePerformance(y_predicted, y_true):

    y_predicted = np.argmax(y_predicted, axis = 1)
    y_true = np.argmax(y_true, axis = 1)


    CM = confusion_matrix(y_true, y_predicted)
    accuracy = (CM[0,0] + CM[1,1])/sum(sum(CM))
    sensitivity = CM[1,1] / (CM[1,1] + CM[1,0])
    specificity = CM[0,0] / (CM[0,0] + CM[0,1])

    precision = CM[1,1] / (CM[1,1] + CM[0,1])
    f1_score = 2 * precision * sensitivity / (precision + sensitivity)
    f2_score = 5 * precision * sensitivity / ((4 * precision) + sensitivity)

    print('Classification accuracy: ', accuracy)
    print('Sensitivity: ', sensitivity)
    print('Specificity: ', specificity)

    print('Precision: ', precision)
    print('F1-score: ', f1_score)
    print('F2-score: ', f2_score)