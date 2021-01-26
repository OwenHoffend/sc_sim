import numpy as np
from PIL import Image

def load_img(path, gs=False):
    """Load an image and return it as a numpy ndarray"""
    image = Image.open(path)
    if gs: #Convert to single-channel greyscale, if desired
        image = image.convert('LA')
        width, height = image.size
        return np.array(image)[0:height, 0:width, 0]
    return np.array(image)

def img_to_bs(img_channel, bs_gen_func, bs_len=255, correlated=True):
    """Convert a single image chnanel into an stochastic bitstream of the specified length.
    bs_gen_func is a bitstream generating function that takes in n (number of bits) and p, 
    the desired probability. If correlated is True, use the same RNG for all pixels"""
    height, width = img_channel.shape
    npb = np.ceil(bs_len / 8.0).astype(np.int) #Compute the number of packed bytes necessary to represent this bitstream
    bs = np.zeros((height, width, npb), dtype=np.uint8) #Initialize nxn array of to hold bit-packed SC bitstreams
    for i in range(height): #Populate the bitstream array
        for j in range(width):
            bs[i][j] = bs_gen_func(bs_len, float(img_channel[i][j]) / 255.0, keep_rng=correlated)
    return bs

def bs_to_img(bs, bs_mean_func):
    """Convert a stochastic bitstream image back into an image"""
    height, width, npb = bs.shape
    img = np.zeros((height, width), dtype=np.uint8)
    for i in range(height): #Populate the bitstream array
        for j in range(width):
            #Default pixel-value encoding is p * 255, might want to try others
            img[i][j] = np.rint(bs_mean_func(bs[i][j]) * 255).astype(np.uint8)
    return img

def img_mse(img1, img2):
    """Compute the MSE between two images.
       Assumes the two images are the same shape"""
    h, w = img1.shape
    return np.sum(np.square(img1 - img2)) / (h * w)

def disp_img_diff(img1, img2):
    """Display the difference between img1 and img2, taken as img2 - img1"""
    diff = img2 - img1
    disp_img(diff)

def save_img(img_arr, path):
    """Save an image from a numpy ndarray at the specified path"""
    Image.fromarray(img_arr).save(path)

def disp_img(img_arr):
    """Display an image from a numpy ndarray (height, width, channels)"""
    image = Image.fromarray(img_arr)
    image.show()

if __name__ == "__main__":
    """Basic test of image loading, manipulation, and storing"""
    lena = load_img("./img/lena256.png")
    lena[0:256, 0:256, 0] = 0 #Zero R and G channels, should produce blue image
    lena[0:256, 0:256, 1] = 0
    disp_img(lena)
    save_img(lena, "./img/lena256_blue.png")