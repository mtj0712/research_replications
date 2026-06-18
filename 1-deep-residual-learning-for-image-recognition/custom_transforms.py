import random
import torch
import torchvision.transforms as transforms

def fancy_pca(img, alpha_std=0.1):
    '''
    This code is largely based on the following: https://github.com/pixelatedbrian/fortnight-furniture/blob/master/src/fancy_pca.py
    '''
    orig_img = img.clone()
    img_flat = orig_img.view(3, -1)
    img_centered = img_flat - torch.mean(img_flat, dim=1, keepdim=True)
    img_cov = torch.cov(img_centered)
    eig_vals, eig_vecs = torch.linalg.eigh(img_cov)
    eig_vals, sort_perm = torch.sort(eig_vals, descending=True)
    eig_vecs = eig_vecs[:, sort_perm]
    m1 = eig_vecs
    m2 = torch.zeros((3, 1), device=img.device, dtype=img.dtype)
    alpha = torch.empty(1, device=img.device, dtype=img.dtype).normal_(0, alpha_std)
    m2[:, 0] = alpha * eig_vals
    add_vect = torch.matmul(m1, m2)
    orig_img += add_vect.view(3, 1, 1)
    orig_img = torch.clamp(orig_img, 0.0, 1.0)

    return orig_img

def random_resize(img):
    return transforms.functional.resize(
        img, random.randint(256, 480)
    )

def subtract_per_pixel_mean(img):
    return img - torch.mean(img, dim=(1, 2), keepdim=True)