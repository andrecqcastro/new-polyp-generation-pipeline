import sys
import os
import numpy as np
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from easydict import EasyDict
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import shift, center_of_mass

# Configuração do caminho e parâmetros
sys.path.append('/content/polyp-GAN/edge-connect/src')

try:
    from models import EdgeModel, InpaintingModel
    from utils import imsave
except ModuleNotFoundError:
    print("O diretório 'src' não foi encontrado ou não contém 'models.py' e 'utils.py'.")

_DEBUG = True

config = EasyDict()
config.PATH_TO_DATA = ""
config.OUTPUT_FOLDER = "/content/drive/MyDrive/Zscan/repolyp/Inpainting/ACPG/dataset/edgeconnect_without_polyp_edge"
config.IMAGE_DIR = "/content/drive/MyDrive/Zscan/repolyp/Inpainting/ACPG/dataset/imagens_removidas"
config.MASK_DIR = "/content/drive/MyDrive/Zscan/repolyp/Inpainting/ACPG/dataset/masks"
config.EDGE_DIR = "/content/drive/MyDrive/Zscan/repolyp/Inpainting/ACPG/dataset/edges"
config.INPUT_IMAGES_DIR = "/content/drive/MyDrive/Zscan/repolyp/Inpainting/ACPG/dataset/images"
config.PATH = '/content/model'
config.DEVICE = 'cpu'
config.MODE = 0
config.N_IMAGES_TO_GENERATE = 8
config.GAN_LOSS = "nsgan"
config.GPU = [0]
config.LR = 0.0001
config.D2G_LR = 0.1
config.BETA1 = 0.0
config.BETA2 = 0.9
config.BATCH_SIZE = 1
config.INPUT_SIZE = 256
config.SIGMA = 2
config.MAX_ITERS = 2e6

# Funções auxiliares
def to_tensor(img):
    return F.to_tensor(Image.fromarray(img)).float()

def cuda(*args):
    return (item.to(config.DEVICE) for item in args)

def resize(img, height=config.INPUT_SIZE, width=config.INPUT_SIZE):
    return np.array(Image.fromarray(img).resize((width, height)))

def postprocess(img):
    if img.dim() > 4:
        img = img.squeeze(0)
    img = img * 255.0
    img = img.permute(0, 2, 3, 1).to(torch.uint8)
    return img

def load_image(path):
    img = np.array(Image.open(path))
    img = resize(img)
    return img, rgb2gray(img)

def load_mask(path):
    mask = np.array(Image.open(path))
    return (resize(mask) > 0).astype(np.uint8) * 255

def load_edge(path):
    return resize(np.array(Image.open(path)))

def align_mask_and_edge_to_position(chosen_mask, chosen_edge, original_mask):
    center_original = center_of_mass(original_mask)
    center_chosen = center_of_mass(chosen_mask)
    shift_y, shift_x = center_original[0] - center_chosen[0], center_original[1] - center_chosen[1]
    aligned_mask = shift(chosen_mask, shift=(shift_y, shift_x), mode='nearest')
    aligned_edge = shift(chosen_edge, shift=(shift_y, shift_x), mode='nearest')
    return aligned_mask, aligned_edge

def load_data(config):
    ip = sorted(glob(os.path.join(config.IMAGE_DIR, '*.jpg')))
    mp = sorted(glob(os.path.join(config.MASK_DIR, '*.jpg')))
    ep = sorted(glob(os.path.join(config.EDGE_DIR, '*.jpg')))
    return ip, mp, ep

# Inicialização do modelo
if __name__ == "__main__":
    inpaint_model = InpaintingModel(config).to(config.DEVICE)
    inpaint_model.load()
    inpaint_model.eval()

    os.makedirs(config.OUTPUT_FOLDER + '/GENERATED_IMAGES', exist_ok=True)
    os.makedirs(config.OUTPUT_FOLDER + '/GENERATED_IMAGES_MASKS', exist_ok=True)

    ip, mp, ep = load_data(config)
    ip, mp, ep = ip[:config.N_IMAGES_TO_GENERATE], mp[:config.N_IMAGES_TO_GENERATE], ep[:config.N_IMAGES_TO_GENERATE]

    for i, (img_p, msk_p, edg_p) in enumerate(zip(ip, mp, ep)):
        img_filename = os.path.basename(img_p)
        
        img, _ = load_image(img_p)
        original_mask = load_mask(msk_p)

        chosen_mask, chosen_image, chosen_edge = align_mask_and_edge_to_position(
            original_mask, load_edge(edg_p), original_mask
        )

        mask = to_tensor(chosen_mask).to(config.DEVICE).unsqueeze(0)
        edge = to_tensor(chosen_edge).to(config.DEVICE).unsqueeze(0)

        img, mask, edge = cuda(to_tensor(img), mask, edge)
        img = img.unsqueeze(0) if img.dim() == 3 else img
        mask = mask.unsqueeze(0) if mask.dim() == 3 else mask
        edge = edge.unsqueeze(0) if edge.dim() == 3 else edge

        if edge.dim() == 5:
            edge = edge.squeeze(0)

        edge = (edge * (1 - (1 - mask)).float()).unsqueeze(0).unsqueeze(0)

        outputs = inpaint_model(img, edge, mask)
        outputs_merged = (outputs * mask) + (img * (1 - mask))

        output = postprocess(outputs_merged)[0]
        imsave(output, os.path.join(config.OUTPUT_FOLDER, 'GENERATED_IMAGES', img_filename))
