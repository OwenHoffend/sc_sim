import sim.bitstreams as bitstreams
import cv.img_io as img_io

def img_to_bs_test(**kwargs):
    """Test image --> bitstream conversion"""
    lena_gs = img_io.load_img("./img/lena256.png", gs=True)
    rng = bitstreams.SC_RNG()
    return img_io.img_to_bs(lena_gs, rng.bs_lfsr, **kwargs)

def img_sc_convert_test(**kwargs):
    """Test image --> bitstream --> image conversion"""
    lena_bs = img_to_bs_test(**kwargs)
    lena_recon = img_io.bs_to_img(lena_bs, bitstreams.bs_mean)
    img_io.disp_img(lena_recon)
    return lena_recon

if __name__ == "__main__":
    #print(img_to_bs_test())
    lena = img_io.load_img("./img/lena256.png", gs=True)
    lena_recon = img_sc_convert_test(bs_len=32, correlated=True)
    print(img_io.img_mse(lena_recon, lena))
    #img_io.disp_img_diff(lena_recon, lena)