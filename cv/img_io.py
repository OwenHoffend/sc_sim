from PIL import Image
from numpy import array

def load_img(path, gs=False):
    """Load an image and return it as a numpy ndarray"""
    image = Image.open(path)
    if gs: #Convert to single-channel greyscale, if desired
        image = image.convert('LA')
        width, height = image.size
        return array(image)[0:height, 0:width, 0]
    return array(image)

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