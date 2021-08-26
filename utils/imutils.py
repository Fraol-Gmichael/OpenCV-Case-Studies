from matplotlib import pyplot as plt
import numpy as np
import cv2
import mahotas


def colored_1D(image, title="Color Histogram", mask=None):

    chans = cv2.split(image)

    if len(chans) == 3:
        colors = ('b', 'g', 'r')
    if len(chans) == 1:
        colors = ('black',)

    figure = plt.figure()
    ax = figure.add_subplot()
    ax.set_title(title)
    ax.set_xlabel("Bins")
    ax.set_ylabel("# of bins")

    for chan, color in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])


def colored_2D(image, mask=None):
    figure = plt.figure()
    num = 131
    chans = cv2.split(image)
    name = [' G and B', ' G and R', ' B and R']
    arr = [[1, 0], [1, 2], [0, 2]]
    for i, each in enumerate(arr):
        ax = figure.add_subplot(num)
        num += 1
        hist = cv2.calcHist([chans[each[0]], chans[each[1]]], [
                            0, 1], mask, [32, 32], [0, 256, 0, 256])

        p = ax.imshow(hist, interpolation='nearest')
        ax.set_title(f'2D Color Histogram for{name[i]}')
        plt.colorbar(p)

    print("2D histogram shape: {}, with {} values".format(
        hist.shape, hist.flatten().shape[0]))


def mask(image, center=None, width=50, rectangle=True):

    mask = np.zeros(image.shape[:2], dtype="uint8")
    center = np.array(center)
    if type(center) == np.array:
        center = [image.shape[1]//2, image.shape[0]//2]
        start = center - width
        end = center + width
    else:
        start = center[0]
        end = center[1]

    if rectangle:
        mask = cv2.rectangle(mask, start, end, 255, -1)
    else:
        mask = cv2.circle(mask, center, width, 255, -1)

    masked = cv2.bitwise_and(image, image, mask=mask)

    return mask, masked


# translate means to shift the image to the left, right, top and bottom
def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return shifted


def rotate(image, angle, center=None, scale=1.0):
    height, width = image.shape[:2]
    if not center:
        center = (width//2, height//2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (width, height))

    return rotated


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    h, w = image.shape[:2]
    if not(width or height):
        return image
    elif width:
        ratio = (width / w)
        dim = (width, int(h*ratio))
    else:
        ratio = (height / h)
        dim = (int(w*ratio), height)

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def crop(image, start=[0, 0], end=None):
    if not end:
        end = [image.shape[1], image.shape[0]]
    cropped = image[start[0]:end[0], start[1]:end[1]]

    return cropped


def arithmetic(image, number, add=False, subtract=False):

    number = np.ones(image.shape, dtype="uint8") * number

    if add and subtract:
        return image
    elif add:
        return cv2.add(image, number)
    elif subtract:
        return cv2.subtract(image, number)
    else:
        return image


def mask2(image, center=None, width=50, rectangle=True):
    mask = np.zeros(image.shape, dtype="uint8")

    if not center:
        center = [image.shape[1]//2, image.shape[0]//2]

    center = np.array(center)

    start = center - width
    end = center + width

    if rectangle:
        masked = cv2.rectangle(mask, start, end, (255, 255, 255), -1)
    else:
        masked = cv2.circle(mask, center, width, (255, 255, 255), -1)

    masked = cv2.bitwise_and(masked, image)

    return masked

    #------------OR-----------#


def hueMaker(image, rgb, hueto='r'):
    (r, g, b) = rgb
    B, G, R = cv2.split(image)
    zeros = np.zeros(image.shape[:2], dtype="uint8")
    hue = 0
    if hueto == 'r':
        hue = cv2.merge([zeros+b, zeros+g, R])
    elif hueto == 'g':
        hue = cv2.merge([zeros+b, G, zeros+r])
    elif hueto == 'b':
        hue = cv2.merge([B, zeros+g, zeros+r])

    return hue


def four_stack(a11, a12, a21, a22):

    row1 = np.hstack([a11, a12])
    row2 = np.hstack([a21, a22])

    return np.vstack([row1, row2])


def simple_tresh(image_org, t_value):
    image = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    (T, thresh) = cv2.threshold(blurred, t_value, 255, cv2.THRESH_BINARY)
    (T, threshInv) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY_INV)
    coins = cv2.bitwise_and(image_org, image_org, mask=threshInv)

    return np.hstack([image_org, coins]), np.hstack([thresh, threshInv])


def flip(image):
    return np.flip(image, 1)


def canny(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    canny = cv2.Canny(image, 150, 255)

    return canny


def sobel(image):
    sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    sobelCombined = cv2.bitwise_or(sobelX, sobelY)

    return sobelCombined


def laplacian(image):
    lap = cv2.Laplacian(image, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))

    return lap


def otsu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    T = mahotas.thresholding.otsu(blurred)
    T = mahotas.thresholding.rc(blurred)

    image = simple_tresh(image, T)

    return image[1]


def counter(image):
    image_drawn = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    edged = cv2.Canny(blurred, 30, 150)
    countours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(image_drawn, countours, -1, (0, 255, 0), 2)

    return countours, image_drawn


def colorTrack(image, lower, upper):
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    mask = cv2.inRange(image, lower, upper)
    colored = cv2.bitwise_and(image, image, mask=mask)

    countours, _ = counter(colored)    

    if len(countours) > 0:
        cnt = sorted(countours, key = cv2.contourArea, reverse=True)[0]
        rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
        cv2.drawContours(image, [rect], -1, (0, 255, 0), 2)

    return image
