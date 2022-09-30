'''
The implementation of the Restore-to-Classify GANs (R2C-GANs).
Author: Mete Ahishali,
Tampere University, Tampere, Finland.

The software implementation is inspired from the following repository: https://github.com/LynnHo/CycleGAN-Tensorflow-2.
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import skimage.io as sio
import numpy as np
import tensorflow as tf

import r2c_gan
import utils
import checkpoints
import data

ap = argparse.ArgumentParser()

ap.add_argument('--datasets_dir', default='dataset/')
ap.add_argument('--experiment_dir', default='output/x_ray_restoration/')
ap.add_argument('--load_size', type=int, default=286) # Load the images with this size.
ap.add_argument('--crop_size', type=int, default=256) # Cropping to this size.
ap.add_argument('--batch_size', type=int, default=32)
ap.add_argument('--q', type=int, default=3) # Order of the operational layer (q parameter).
ap.add_argument('--saveImages', default = 'False', help = 'To save the restored images.')
ap.add_argument('--method', help='operational, convolutional, convolutional-light', default='operational') # Type of the transformation in R2C-GANs.
args = vars(ap.parse_args())

# Loading data.
A_img_paths_test, A_label_test = utils.readData(args['datasets_dir'] + 'testA' + '/*.png')

A_dataset_test = data.make_dataset(A_img_paths_test, A_label_test,
                                    args['batch_size'], args['load_size'], args['crop_size'],
                                    training=False, drop_remainder=False,
                                    shuffle=False, repeat=1)

# Creating models.
r2c_gan = r2c_gan.r2c_gan()
r2c_gan.filter = args['method']
r2c_gan.set_G_A2B(input_shape=(args['crop_size'], args['crop_size'], 3), q = args['q'])

# Restore the checkpoint.
checkDir = 'output/checkpoints/' + args['method']
if not os.path.exists(checkDir): os.makedirs(checkDir)
checkpoints.Checkpoint(dict(G_A2B=r2c_gan.G_A2B), checkDir).restore()

@tf.function
def sample_A2B(A):
    A2B, y_pred = r2c_gan.G_A2B(A, training=False)
    return A2B, y_pred

# Classification and restoration:
if args['saveImages'] == 'True':
    save_dir = 'output/samples_testing/'
    if not os.path.exists(save_dir): os.makedirs(save_dir)

y_predicted = []
y_true = []
i = 0
# Loop through the test set.
for A in A_dataset_test:
    A2B, y_pred = sample_A2B(A[0])

    if args['saveImages'] == 'True':
        A2B_II, _ = sample_A2B(A2B) # Second iteration.
        A2B_III, _ = sample_A2B(A2B_II) # Third iteration.
    
        for A_i, A2B_i, A2B_ii, A2B_iii, y_pred_i, y_true_i in zip(A[0], A2B, A2B_II, A2B_III, y_pred, A[1]):

            img = np.concatenate([A_i.numpy(), A2B_i.numpy(), A2B_ii.numpy(), A2B_iii.numpy()], axis=1)
            sio.imsave(save_dir + A_img_paths_test[i].split('/')[-1],
                        ((img + 1.) / 2. * 255).astype(np.uint8))

            y_predicted.append(y_pred_i.numpy())
            y_true.append(y_true_i.numpy())
            i += 1
    else:
        for y_pred_i, y_true_i in zip(y_pred, A[1]):
            y_predicted.append(y_pred_i.numpy())
            y_true.append(y_true_i.numpy())
            i += 1

print('Processed number of images: ', i)

utils.computePerformance(y_predicted, y_true)