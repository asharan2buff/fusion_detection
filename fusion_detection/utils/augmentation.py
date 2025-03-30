# utils/augmentation.py
import cv2
import numpy as np
import random

def salt_pepper_noise(image, s_vs_p=0.5, amount=0.04):
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords[0], coords[1], :] = 255

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords[0], coords[1], :] = 0
    return out

def pixelate(image, pixel_size=8):
    h, w = image.shape[:2]
    temp = cv2.resize(image, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def gaussian_blur(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_augmentations(image, augmentations=None):
    if augmentations is None:
        return image
    
    result = image.copy()
    
    for aug in augmentations:
        if aug == 'SaltPapperNoise':
            result = salt_pepper_noise(result)
        elif aug == 'pixelate':
            result = pixelate(result)
        elif aug == 'GaussianBlur':
            result = gaussian_blur(result)
    
    return result
