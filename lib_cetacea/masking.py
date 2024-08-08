import tensorflow as tf

from tensorflow.keras.layers import * 
from lib_cetacea.patches import *


def rand_idx(batch_size, n_patches, mask_percentage):
    num_mask = int(n_patches * mask_percentage)
    rand_indices = tf.argsort(
        tf.random.uniform(shape=(batch_size, n_patches)), axis=-1
    )
    mask_indices = rand_indices[:, : num_mask]
    unmask_indices = rand_indices[:, num_mask :]
    return mask_indices, unmask_indices, num_mask


def generate_masked_image(patch, unmask_index):
    new_patch = np.zeros_like(patch)    
    print(unmask_index.shape, new_patch.shape)
    for i in unmask_index:
        new_patch[i] = patch[i]
    return new_patch


class MaskedPatchEncoder(Layer):

    def __init__(self, patch_size, mask_percentage, max_len=10000, **kwargs):
        super().__init__(**kwargs)
        self.mask_percentage = mask_percentage
        self.mask_token = tf.Variable(
            tf.random.normal([1, patch_size * patch_size]), trainable=True
        )
        self.position_embedding = Embedding(
            input_dim=max_len, output_dim=patch_size * patch_size
        )

        
    def call(self, x):
        batch_size, n_patches, dim = x.shape
        mask_indices, unmask_indices, num_mask = rand_idx(batch_size, n_patches, self.mask_percentage)
        
        positions = tf.range(start=0, limit=n_patches, delta=1)        
        pos_embeddings = self.position_embedding(positions[tf.newaxis, ...])
        pos_embeddings = tf.tile(
            pos_embeddings, [batch_size, 1, 1]
        )
        unmasked_positions = tf.gather(
            pos_embeddings, unmask_indices, axis=1, batch_dims=1
        ) 
        masked_positions = tf.gather(
            pos_embeddings, mask_indices, axis=1, batch_dims=1
        )  

        
        unmasked_embeddings = tf.gather(
            x, unmask_indices, axis=1, batch_dims=1
        )
        unmasked_embeddings = unmasked_embeddings + unmasked_positions

        masked_embeddings = tf.repeat(self.mask_token, repeats=num_mask, axis=0)
        masked_embeddings = tf.repeat(
            masked_embeddings[tf.newaxis, ...], repeats=batch_size, axis=0
        )
        masked_embeddings = masked_embeddings + masked_positions
        
        return (
            unmasked_embeddings,
            masked_embeddings,
            unmasked_positions,
            mask_indices,
            unmask_indices,
        )        
        return x 
