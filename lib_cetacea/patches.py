import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt


class Patches(tf.keras.layers.Layer):
     def __init__(self, patch_size=10, **kwargs):
         super().__init__(**kwargs)
         self.patch_size = patch_size
         self.resize = tf.keras.layers.Reshape((-1, patch_size * patch_size))

     def call(self, images):
         patches = tf.image.extract_patches(
             images=images,
             sizes=[1, self.patch_size, self.patch_size, 1],
             strides=[1, self.patch_size, self.patch_size, 1],
             rates=[1, 1, 1, 1],
             padding="SAME",)
         patches = self.resize(patches)
         return patches


def plot_grid(filename, tensor, n_patches_w, n_patches_h, patch_size):
     n_patches_w, n_patches_h = n_patches_w // patch_size, n_patches_h // patch_size

     tensor = tensor[0]
     vmin = np.min(tensor)
     vmax = np.max(tensor)
     
     plt.figure(figsize=(n_patches_h, n_patches_w))
     k = 1
     for i in range(n_patches_w):
          for j in range(n_patches_h):
               plt.subplot(n_patches_w, n_patches_h, k)               
               if k < len(tensor):
                    plt.imshow(tf.reshape(tensor[k - 1, :], (patch_size, patch_size)), vmin=vmin, vmax=vmax)               
               plt.axis("off")
               k += 1
     plt.savefig(filename, bbox_inches='tight')


def reconstruct_from_patch(patch, patch_size, img_w):
     n = img_w // patch_size 
     num_patches = patch[0].shape[0]
     patch = tf.reshape(patch, (num_patches, patch_size, patch_size, 1))
     rows = tf.split(patch, n, axis=0)
     rows = [tf.concat(tf.unstack(x), axis=1) for x in rows]
     reconstructed = tf.concat(rows, axis=0)
     return reconstructed
