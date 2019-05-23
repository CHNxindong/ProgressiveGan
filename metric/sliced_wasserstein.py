# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import scipy.ndimage

#----------------------------------------------------------------------------

def get_descriptors_for_minibatch(minibatch, nhood_size, nhoods_per_image):
    S = minibatch.shape # (minibatch, channel, height, width)
    assert len(S) == 4 and S[1] == 3
    N = nhoods_per_image * S[0]
    H = nhood_size // 2
    nhood, chan, x, y = np.ogrid[0:N, 0:3, -H:H+1, -H:H+1]
    img = nhood // nhoods_per_image
    x = x + np.random.randint(H, S[3] - H, size=(N, 1, 1, 1))
    y = y + np.random.randint(H, S[2] - H, size=(N, 1, 1, 1))
    idx = ((img * S[1] + chan) * S[2] + y) * S[3] + x
    return minibatch.flat[idx]

#----------------------------------------------------------------------------

def finalize_descriptors(desc):
    if isinstance(desc, list):
        desc = np.concatenate(desc, axis=0)
    assert desc.ndim == 4 # (neighborhood, channel, height, width)
    desc -= np.mean(desc, axis=(0, 2, 3), keepdims=True)
    desc /= np.std(desc, axis=(0, 2, 3), keepdims=True)
    desc = desc.reshape(desc.shape[0], -1)
    return desc

#----------------------------------------------------------------------------

def sliced_wasserstein(A, B, dir_repeats, dirs_per_repeat):
    assert A.ndim == 2 and A.shape == B.shape                           # (neighborhood, descriptor_component)
    results = []
    for repeat in range(dir_repeats):
        dirs = np.random.randn(A.shape[1], dirs_per_repeat)             # (descriptor_component, direction)
        dirs /= np.sqrt(np.sum(np.square(dirs), axis=0, keepdims=True)) # normalize descriptor components for each direction
        dirs = dirs.astype(np.float32)
        projA = np.matmul(A, dirs)                                      # (neighborhood, direction)
        projB = np.matmul(B, dirs)
        projA = np.sort(projA, axis=0)                                  # sort neighborhood projections for each direction
        projB = np.sort(projB, axis=0)
        dists = np.abs(projA - projB)                                   # pointwise wasserstein distances
        results.append(np.mean(dists))                                  # average over neighborhoods and directions
    return np.mean(results)                                             # average over repeats

#----------------------------------------------------------------------------

def downscale_minibatch(minibatch, lod):
    if lod == 0:
        return minibatch
    t = minibatch.astype(np.float32)
    for i in range(lod):
        t = (t[:, :, 0::2, 0::2] + t[:, :, 0::2, 1::2] + t[:, :, 1::2, 0::2] + t[:, :, 1::2, 1::2]) * 0.25
    return np.round(t).clip(0, 255).astype(np.uint8)

#----------------------------------------------------------------------------

gaussian_filter = np.float32([
    [1, 4,  6,  4,  1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4,  6,  4,  1]]) / 256.0

def pyr_down(minibatch): # matches cv2.pyrDown()
    assert minibatch.ndim == 4
    return scipy.ndimage.convolve(minibatch, gaussian_filter[np.newaxis, np.newaxis, :, :], mode='mirror')[:, :, ::2, ::2]

def pyr_up(minibatch): # matches cv2.pyrUp()
    assert minibatch.ndim == 4
    S = minibatch.shape
    res = np.zeros((S[0], S[1], S[2] * 2, S[3] * 2), minibatch.dtype)
    res[:, :, ::2, ::2] = minibatch
    return scipy.ndimage.convolve(res, gaussian_filter[np.newaxis, np.newaxis, :, :] * 4.0, mode='mirror')

def generate_laplacian_pyramid(minibatch, num_levels):
    pyramid = [np.float32(minibatch)]
    for i in range(1, num_levels):
        pyramid.append(pyr_down(pyramid[-1]))
        pyramid[-2] -= pyr_up(pyramid[-1])
    return pyramid

def reconstruct_laplacian_pyramid(pyramid):
    minibatch = pyramid[-1]
    for level in pyramid[-2::-1]:
        minibatch = pyr_up(minibatch) + level
    return minibatch

#----------------------------------------------------------------------------

class API:
    def __init__(self, image_shape):
        self.nhood_size         = 7
        self.nhoods_per_image   = 128
        self.dir_repeats        = 4
        self.dirs_per_repeat    = 128
        self.resolutions = []
        self.descriptors={}
        res = image_shape[1]
        while res >= 16:
            self.resolutions.append(res)
            res //= 2

    def get_metric_names(self):
        return ['SWDx1e3_%d' % res for res in self.resolutions] + ['SWDx1e3_avg']

    def get_metric_formatting(self):
        return ['%-13.4f'] * len(self.get_metric_names())

    def begin(self, mode):
        assert mode in ['warmup', 'reals', 'fakes']
        self.descriptors[mode] = [[] for res in self.resolutions]

    def feed(self, mode, minibatch):
        for lod, level in enumerate(generate_laplacian_pyramid(minibatch, len(self.resolutions))):
            desc = get_descriptors_for_minibatch(level, self.nhood_size, self.nhoods_per_image)
            self.descriptors[mode][lod].append(desc)

    def end(self):
        desc_reals = [finalize_descriptors(d) for d in self.descriptors["reals"]]
        desc_fakes = [finalize_descriptors(d) for d in self.descriptors["fakes"]]
        del self.descriptors
        dist = [sliced_wasserstein(dreal, dfake, self.dir_repeats, self.dirs_per_repeat) for dreal, dfake in zip(desc_reals, desc_fakes)]
        del desc_reals
        del desc_fakes
        dist = [d * 1e3 for d in dist] # multiply by 10^3
        return dist + [np.mean(dist)]

#----------------------------------------------------------------------------
from scipy.misc import imread
def get_images_np(files):
    return np.concatenate(np.array([np.expand_dims(imread(str(fn)).astype(np.float32), 0) for fn in files]),0).transpose(0,3,1,2)
if __name__=="__main__":
    import glob

    from tqdm import tqdm
    api=API(image_shape=[256,256,3])
    api.begin("fakes")
    path = glob.glob("/home/guyuchao/ssd/PycharmProjects/testgan/generate/*.jpg")
    for i in tqdm(range(25)):
        api.feed("fakes",get_images_np(path[i*100:(i+1)*100]))

    api.begin("reals")
    path = glob.glob("/home/guyuchao/ssd/PycharmProjects/testgan/real/*.jpg")
    for i in tqdm(range(25)):
        api.feed("reals", get_images_np(path[i * 100:(i + 1) * 100]))
    print(api.end())#[0.092510438224158206]