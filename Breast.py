from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
import numpy as np
from keras.models import load_model
from werkzeug.utils import secure_filename
import joblib
from pydicom import dcmread
import os


# Cargamos el modelo preentrenado
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'CNN.h5')
modelo = load_model(model_path, compile=False)

clasificador_path = os.path.join(base_path, 'Clasificador.sav')
CLASIFICADOR = joblib.load(clasificador_path)



from scipy.fftpack import fftshift, ifftshift
from scipy.fftpack import fft2, ifft2

def lowpassfilter(size, cutoff, n):
    if cutoff < 0. or cutoff > 0.5:
        raise Exception('cutoff must be between 0 and 0.5')
    elif n % 1:
        raise Exception('n must be an integer >= 1')
    if len(size) == 1:
        rows = cols = size
    else:
        rows, cols = size

    if (cols % 2):
        xvals = np.arange(-(cols - 1) / 2.,
                          ((cols - 1) / 2.) + 1) / float(cols - 1)
    else:
        xvals = np.arange(-cols / 2., cols / 2.) / float(cols)

    if (rows % 2):
        yvals = np.arange(-(rows - 1) / 2.,
                          ((rows - 1) / 2.) + 1) / float(rows - 1)
    else:
        yvals = np.arange(-rows / 2., rows / 2.) / float(rows)

    x, y = np.meshgrid(xvals, yvals, sparse=True)
    radius = np.sqrt(x * x + y * y)

    return ifftshift(1. / (1. + (radius / cutoff) ** (2. * n)))


def rayleighmode(data, nbins=50):
    n, edges = np.histogram(data, nbins)
    ind = np.argmax(n)
    return (edges[ind] + edges[ind + 1]) / 2.


def perfft2(im, compute_P=True, compute_spatial=False):

    if im.dtype not in ['float32', 'float64']:
        im = np.float64(im)

    rows, cols = im.shape
    s = np.zeros_like(im)
    s[0, :] = im[0, :] - im[-1, :]
    s[-1, :] = -s[0, :]
    s[:, 0] = s[:, 0] + im[:, 0] - im[:, -1]
    s[:, -1] = s[:, -1] - im[:, 0] + im[:, -1]

    x, y = (2 * np.pi * np.arange(0, v) / float(v) for v in (cols, rows))
    cx, cy = np.meshgrid(x, y)

    denom = (2. * (2. - np.cos(cx) - np.cos(cy)))
    denom[0, 0] = 1.     

    S = fft2(s) / denom
    S[0, 0] = 0      
    
    if compute_P or compute_spatial:

        P = fft2(im) - S

        if compute_spatial:
            s = ifft2(S).real
            p = im - s

            return S, P, s, p
        else:
            return S, P
    else:
        return S


def filtergrid(rows, cols):

    
    u1, u2 = np.meshgrid(np.linspace(-0.5, 0.5, cols, endpoint=(cols % 2)),
                         np.linspace(-0.5, 0.5, rows, endpoint=(rows % 2)),
                         sparse=True)

    
    u1 = ifftshift(u1)
    u2 = ifftshift(u2)

    
    radius = np.sqrt(u1 * u1 + u2 * u2)

    return radius, u1, u2


def Riesz2D(img):


    nscale=5 
    minWaveLength=3 
    mult=2.1 
    sigmaOnf=0.55
    k=2 
    cutOff=0.5 
    g=10 
    noiseMethod=-1 
    deviationGain=1.5
    
    if img.dtype not in ['float32', 'float64']:
        img = np.float64(img)
        imgdtype = 'float64'
    else:
        imgdtype = img.dtype

    if img.ndim == 3:
        img = img.mean(2)

    rows, cols = img.shape

    epsilon = 1E-4          
    _, IM = perfft2(img)     

    zeromat = np.zeros((rows, cols), dtype=imgdtype)
    sumAn = zeromat.copy()
    sumf = zeromat.copy()
    sumh1 = zeromat.copy()
    sumh2 = zeromat.copy()

    radius, u1, u2 = filtergrid(rows, cols)

    radius[0, 0] = 1.

    H = (1j * u1 - u2) / radius

    
    lp = lowpassfilter((rows, cols), .45, 15)
    logGaborDenom = 2. * np.log(sigmaOnf) ** 2.

    for ss in range(nscale):
        wavelength = minWaveLength * mult ** ss
        fo = 1. / wavelength  
        logRadOverFo = (np.log(radius / fo))
        logGabor = np.exp(-(logRadOverFo * logRadOverFo) / logGaborDenom)
        logGabor *= lp      
        logGabor[0, 0] = 0.  

        IMF = IM * logGabor   
        f = np.real(ifft2(IMF))  

        h = ifft2(IMF * H)
        h1, h2 = np.real(h), np.imag(h)

        
        An = np.sqrt(f * f + h1 * h1 + h2 * h2)

        
        sumAn += An
        sumf += f
        sumh1 += h1
        sumh2 += h2

        if ss == 0:
            
            if noiseMethod == -1:
                tau = np.median(sumAn.flatten()) / np.sqrt(np.log(4))

            maxAn = An
        else:
            
            
            maxAn = np.maximum(maxAn, An)


        width = (sumAn / (maxAn + epsilon) - 1.) / (nscale - 1)

        
        
        weight = 1. / (1. + np.exp(g * (cutOff - width)))

        if noiseMethod >= 0:
            T = noiseMethod

        
        else:
            totalTau = tau * (1. - (1. / mult) ** nscale) / (1. - (1. / mult))

            
            EstNoiseEnergyMean = totalTau * np.sqrt(np.pi / 2.)
            EstNoiseEnergySigma = totalTau * np.sqrt((4 - np.pi) / 2.)

            
            T = np.max((EstNoiseEnergyMean + k * EstNoiseEnergySigma, epsilon))

    
    ori = np.arctan(-sumh2 / sumh1)

    
    ori = np.fix((ori % np.pi) / np.pi * 180.)

    
    ft = np.arctan2(sumf, np.sqrt(sumh1 * sumh1 + sumh2 * sumh2))

    
    energy = np.sqrt(sumf * sumf + sumh1 * sumh1 + sumh2 * sumh2)

    phase_dev = np.maximum(
        1. - deviationGain * np.arccos(energy / (sumAn + epsilon)), 0)
    energy_thresh = np.maximum(energy - T, 0)

    M = weight * phase_dev * energy_thresh / (energy + epsilon)

    return M, ori, ft

def Stats_CNN(Desc):
    Desc = np.array(Desc)
    Var   = np.var(Desc)
    Mean  = np.mean(Desc)
    Std   = np.std(Desc)
    Max   = np.max(Desc)
    Min   = np.min(Desc)
    Stats = np.hstack((Var, Mean, Std, Max, Min))
    return Stats


def cropBorders(img, l=0.01, r=0.01, u=0.04, d=0.04):
    nrows, ncols = img.shape
    
    l_crop = int(ncols * l)
    r_crop = int(ncols * (1 - r))
    u_crop = int(nrows * u)
    d_crop = int(nrows * (1 - d))
    cropped_img = img[u_crop:d_crop, l_crop:r_crop]
    return cropped_img

def clahe(img, clip=2.0, tile=(8, 8)):

    img = cv2.normalize(img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F,)
    img_uint8 = img.astype("uint8")
    clahe_create = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    clahe_img = clahe_create.apply(img_uint8)

    return clahe_img

def preproceso(image):

    dim = (632, 512) 
    image = cv2.resize(image, dim)

    image[np.where(image == 255)] = 0 
    image[np.where((image >= 1) & (image <= 25))] = 0
    image = cropBorders(image, l=0.05, r=0.05, u=0.05, d=0.05)
    image = clahe(image)
    m = cv2.moments(image)
    i = np.int32(m['m10']//m['m00'])
    j = np.int32(m['m01']//m['m00'])
    centro = (int(i),int(j))
    cropped = cv2.getRectSubPix(image, (256,256), centro)
    return cropped



def elegir_imagen():
    
    path_image = filedialog.askopenfilename(filetypes = [
        ("image", ".jpeg"),
        ("image", ".png"),
        ("image", ".jpg")])
    if len(path_image) > 0:
        global image
        
        image = cv2.imread(path_image, 0)
        imageToShow= imutils.resize(image, width=180)
        im = Image.fromarray(imageToShow )
        img = ImageTk.PhotoImage(image=im)
        lblInputImage.configure(image=img)
        lblInputImage.image = img
        
        
        lblInfo1 = Label(root, text="Mastografía:", font="bold")
        lblInfo1.grid(column=0, row=1, padx=5, pady=5)

        
def Clasificacion():
    global image

    imageToShowOutput = preproceso(image)
    
    
    im = Image.fromarray(imageToShowOutput)
    img = ImageTk.PhotoImage(image=im)
    lblOutputImage.configure(image=img)
    lblOutputImage.image = img

    
    lblInfo3 = Label(root, text="Región de interés:", font="bold")
    lblInfo3.grid(column=1, row=0, padx=5, pady=5)

    orig1, orig2, orig3 = Riesz2D(imageToShowOutput)
    imagen  = cv2.merge([orig1, orig2, orig3])


    dims = list(modelo.input_shape[1:4]) 
    imagen = cv2.resize(imagen, (dims[0], dims[1])).reshape(-1, dims[0], dims[1], dims[2]) 

    features = modelo.predict(imagen, verbose = 0)                   
    dim_feat = list(features.shape) 
    vector_features = dim_feat[0] * dim_feat[1]  
    features = features.reshape(1, vector_features)
    stats = Stats_CNN(features)
    stats = stats.reshape(1, 5)
    preds = CLASIFICADOR.predict(stats)


    if preds == 0:
        result = 'NORMAL'
    elif preds == 1:
        result = 'CANCER'

    result = str(result) 

    
    lblInfo4 = Label(root, text= result , font="bold")
    lblInfo4.grid(column=1, row=5, padx=5, pady=5)

if __name__ == '__main__':

    
    root = Tk()
    root.title("Clasificación de cáncer de mama")

    
    btn = Button(root, text="Elegir imagen", width=25, font="bold", command=elegir_imagen)
    btn.grid(column=0, row=0, padx=5, pady=5)

    
    lblOutputImage = Label(root)
    lblOutputImage.grid(column=1, row=1, rowspan=6)


    
    lblInputImage = Label(root)
    lblInputImage.grid(column=0, row=2)


    
    lblInfo2 = Label(root, text="Realiza tu predicción", font="bold", width=25)
    lblInfo2.grid(column=0, row=3, padx=5, pady=5)

    
    btn = Button(root, text="Diagnóstico", font="bold", width=25, command= Clasificacion)
    btn.grid(column=0, row=5, padx=5, pady=5)


    root.mainloop()

