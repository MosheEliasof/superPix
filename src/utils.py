import kornia
import torch.nn.functional as F
import torch
import numpy as np


class UnNormalize(object):
    def __init__(self, mean, std,asnumpy=False):
        self.mean = mean
        self.std = std
        self.asnumpy=asnumpy

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        
        if self.asnumpy:
            return tensor.squeeze().transpose(0, 2).transpose(0, 1)
        return tensor

# Losses
def mutual_information(logits, coeff):
        prob = logits.softmax(1)
        pixel_wise_ent = - (prob * F.log_softmax(logits, 1)).sum(1).mean()
        marginal_prob = prob.mean((2, 3))
        marginal_ent = - (marginal_prob *
                          torch.log(marginal_prob + 1e-16)).sum(1).mean()
        return pixel_wise_ent - coeff * marginal_ent

def TV_smoothness(logits, image):
        prob = logits.softmax(1)
        dp_dx = prob[..., :-1] - prob[..., 1:]
        dp_dy = prob[..., :-1, :] - prob[..., 1:, :]
        di_dx = image[..., :-1] - image[..., 1:]
        di_dy = image[..., :-1, :] - image[..., 1:, :]
        di_dx2 = 0.5 * ((di_dx[:, :, :-1, :] ** 2) + (di_dx[:, :, 1:, :] ** 2))
        di_dy2 = 0.5 * ((di_dy[:, :, :, :-1] ** 2) + (di_dy[:, :, :, 1:] ** 2))

        dp_dx2 = 0.5 * ((dp_dx[:, :, :-1, :] ** 2) + (dp_dx[:, :, 1:, :] ** 2))
        dp_dy2 = 0.5 * ((dp_dy[:, :, :, :-1] ** 2) + (dp_dy[:, :, :, 1:] ** 2))

        TV_image = torch.sqrt(di_dx2 + di_dy2 + 1e-8).sum(1)
        TV_prob = torch.sqrt(dp_dx2 + dp_dy2 + 1e-8).sum(1)
        TV = (TV_image * TV_prob).mean()
        return TV

def smoothness(logits, image):
    prob = logits.softmax(1)
    dp_dx = prob[..., :-1] - prob[..., 1:]
    dp_dy = prob[..., :-1, :] - prob[..., 1:, :]
    di_dx = image[..., :-1] - image[..., 1:]
    di_dy = image[..., :-1, :] - image[..., 1:, :]

    return (dp_dx.abs().sum(1) * (-di_dx.pow(2).sum(1) / 8).exp()).mean() + \
            (dp_dy.abs().sum(1) * (-di_dy.pow(2).sum(1) / 8).exp()).mean()

def ContourLoss(mean_img, image):
    lap_mean_img = kornia.filters.laplacian(mean_img, 3)
    lap_mean_img = F.softmax(lap_mean_img / lap_mean_img.abs().max(),dim=-1)

    lap_img = kornia.filters.laplacian(image, 3)
    lap_img = F.softmax(lap_img / lap_img.abs().max(),dim=-1)

    return F.kl_div(lap_mean_img, lap_img,reduction='batchmean')

def reconstruction(recons, image):
    return F.mse_loss(recons, image)

def imageFromSpix(spix, input):
        probs = F.softmax(spix, 1)
        input_img = input[:, :3, :, :]
        probs = probs.unsqueeze(0)
        votes = input_img[:, :, None, :, :] * probs
        vals = (votes.sum((3, 4)) / probs.sum((3, 4))
                ).unsqueeze(-1).unsqueeze(-1)
        mean_img = (vals * probs[:, None, :, :, :]).sum(3).squeeze(0)

        return mean_img

def imageFromHardSpix(spix, image):
    mean_img = np.zeros_like(image)
    for ii in np.unique(spix):
        mean_val = image[spix == ii, :].mean(dim=0)
        mean_img[spix == ii] = mean_val

    return mean_img