import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import numpy as np
from ASPP import ASPP_SuperPix

from utils import UnNormalize, imageFromHardSpix

def getParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="./img2.jfif", type=str, help="/path/to/image")
    parser.add_argument("--n_spix", default=100, type=int, help="number of superpixels")
    parser.add_argument("--n_filters", default=32, type=int, help="number of convolution filters")
    parser.add_argument("--n_layers", default=3, type=int, help="number of convolution layers")
    parser.add_argument("--lam", default=2, type=float, help="coefficient of marginal entropy")
    parser.add_argument("--alpha", default=2, type=float, help="coefficient of smoothness loss")
    parser.add_argument("--beta", default=2, type=float, help="coefficient of reconstruction loss")
    parser.add_argument("--lr", default=1e-2, type=float, help="learning rate")
    parser.add_argument("--n_iter", default=100, type=int, help="number of iterations")
    parser.add_argument("--out_dir", default="./", type=str, help="output directory")
    parser.add_argument("--img_size", default=128, type=int, help="number of iterations")
    parser.add_argument("--use_recons", default=True, type=bool, help="if to optimize also for reconstruction")
    parser.add_argument("--use_spix_recons", default=True, type=bool,
                        help="if previous is true, then can choose to recons the spix")
    parser.add_argument("--useTV", default=False, type=bool,
                        help="if to use TV smoothness (if false use L2 smoothness)")
    parser.add_argument("--cuda", default="cuda:0", type=str, help="Which GPU to use, or cpu")                    

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = getParser()
    device = args.cuda if torch.cuda.is_available() else "cpu"

    # Create Model
    model = ASPP_SuperPix(atrous_rates=[2, 4, 8], in_c=5, n_spix=args.n_spix, n_filters=args.n_filters,
                          n_layers=args.n_layers,
                          image_size=args.img_size, use_recons=args.use_recons,
                          useSuperPixRecons=args.use_spix_recons, useTV=args.useTV).to(device)

    # Load and resize image
    img = plt.imread(args.image)
    img = torch.from_numpy(np.copy(img)).permute(2, 0, 1).unsqueeze(0) / 255
    img = F.interpolate(img.float(), size=args.img_size, mode='bilinear', align_corners=False)
    img = img.squeeze().permute(1, 2, 0)

    # Optimize the SuperPixels
    spix, an_img, input_tensor, mean, std, recons = model.optimize(img,
                                                                     args.n_iter,
                                                                     args.lr,
                                                                     args.lam,
                                                                     args.alpha,
                                                                     args.beta,
                                                                     device)
    un_norm = UnNormalize(mean,std,asnumpy=True)
    input_tensor = np.clip(un_norm(input_tensor),a_max=1,a_min=0)
    mean_img = un_norm(mean_img)
    recons_img = un_norm(recons)

    mean_img = imageFromHardSpix(spix, img)


    # Display image
    plt.rcParams["figure.figsize"] = (50,50)
    fig, ax = plt.subplots(1, 3, )
    ax[0].imshow(mark_boundaries(img.numpy(), spix))
    ax[0].set_title('Spix')
    ax[1].imshow(mean_img)
    ax[1].set_title('Spix Image')
    ax[2].imshow(input_tensor)
    ax[2].set_title('GT Image')
    _=plt.axis("off")
    plt.show()

    plt.savefig('output.png',dpi=200)