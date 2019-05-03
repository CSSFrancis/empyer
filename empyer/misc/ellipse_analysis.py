import numpy as np
import math
import random
import matplotlib.pyplot as plt
from numpy.linalg import eig, inv
from collections import namedtuple
from matplotlib.patches import Ellipse


def find_center(img):
    """
    Function which finds the center of a diffraction pattern

    Using an algorithm described in T.C. Patterson et al. Ultramicroscopy 103, 275 (2005). The algorithm assumes the
    sample is circularly symmetric.Takes a random sampling of positions in a cropped region and compares to other points

    Parameters
    ---------
    img: array-like
        An input image with an imposed circle of higher intensity
    """
    def mean_squared_displacement(iterations, mask_x, mask_y, center, img, no_mask=True):
        count = 0
        msd = 0
        points = len(mask_x)
        for i in range(iterations):
            random_position = random.randint(0, points - 1)
            initial_x, initial_y = mask_x[random_position], mask_y[random_position]
            radius = calculate_magnitude(mask_x[random_position], mask_y[random_position])
            theta = random.uniform(0, 2 * np.pi)
            x_rotated, y_rotated = polar_to_cartesian(theta, radius, center)
            initial_x = int(initial_x + center[0])
            initial_y = int(initial_y + center[1])
            final, initial = img[x_rotated][y_rotated], img[initial_x][initial_y]

            if no_mask or final.mask is True or initial.mask is True:
                msd = msd + (final - initial) ** 2
                count = count + 1

        if count:
            msd = msd / count
        else:
            msd = 0
        return msd

    def create_circular_grid(inner_radius, outer_radius):
        x, y = np.meshgrid(range(-outer_radius, outer_radius + 1), range(-outer_radius, outer_radius + 1))
        r = np.sqrt(x ** 2 + y ** 2)
        inside = r < outer_radius
        x, y = x[inside], y[inside]
        return x, y

    def calculate_magnitude(x, y):
        magnitude = np.sqrt((x) ** 2 + (y) ** 2)
        return magnitude

    def polar_to_cartesian(theta, r, center):
        # could do linear interpolation but way more computationally difficult
        x = int(math.floor(r * np.cos(theta)) + center[0])
        y = int(math.floor(r * np.sin(theta)) + center[1])
        return x, y

    search_radius = 7
    all_msd = []
    shape = np.shape(img)
    image_center = np.divide(shape, 2)
    left_bound, right_bound = int(image_center[0]-(search_radius*4)), int(image_center[1]+(search_radius*4))
    MSD = namedtuple('MSD', 'mean_displacement position')
    for x in range(left_bound, right_bound, 4):
        for y in range(left_bound, right_bound, 4):
            center = [x, y]
            mask_x, mask_y = create_circular_grid(50,150)
            msd = mean_squared_displacement(500, mask_y, mask_y, center, img)
            msd =MSD(mean_displacement=msd, position=[x, y])
            all_msd.append(msd)
    center = min(all_msd, key=lambda k: k.mean_displacement).position
    msd2 = []
    left_bound, right_bound = int(center[0]-10), int(center[1]+10)
    for x in range(left_bound, right_bound, 2):
        for y in range(left_bound, right_bound, 2):
            center = [x, y]
            mask_x, mask_y = create_circular_grid(500, 150)
            msd = mean_squared_displacement(500, mask_y, mask_y, center, img)
            msd =MSD(mean_displacement=msd, position=[x, y])
            msd2.append(msd)
    center = min(msd2, key=lambda k: k.mean_displacement).position
    return center


def solve_ellipse(img, mask=None, interactive=False, num_points=500, plot=False):
    """Takes a 2-d array image and allows you to solve for the equivalent ellipse.

    Parameters
    ----------
    img : array-like
        Image with ellipse with more intense
    interactive: bool
        Allows you to pick points for the ellipse instead of taking the top 2000 points
    plot: bool
        plots the unwrapped image as well as a super imposed ellipse
    Returns
    ----------
    center: array-like
        In cartesian coordinates or (x,y)!!!!!! arrays are in y,x
    lengths:array-like
        The 'length' in pixels of the major and minor axis
    angle:float
        In radians based on the major axis
    """

    def fit_ellipse(x,y):
        x = x[:, np.newaxis]  # reshaping the x and y axis
        y = y[:, np.newaxis]
        D = np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
        S = np.dot(D.T, D)
        C = np.zeros([6,6])
        C[0, 2] = C[2, 0] = 2; C[1, 1] = -1
        E, V = eig(np.dot(inv(S), C))
        n = np.argmax(np.abs(E))
        a = V[:, n]
        return a

    def ellipse_center(a):
        b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
        num = b * b - a * c
        x0 = (c * d - b * f) / num
        y0 = (a * f - b * d) / num
        return np.array([x0, y0])

    def ellipse_axis_length(a):
        b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
        up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
        down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        res1 = np.sqrt(up / down1)
        res2 = np.sqrt(up / down2)
        return np.array([res1, res2])

    def ellipse_angle_of_rotation( a ):
        b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        if b == 0:
            if a > c:
                return 0
            else:
                return np.pi/2
        else:
            if a > c:
                return np.arctan(2*b/(a-c))/2
            else:
                return np.pi/2 + np.arctan(2*b/(a-c))/2
    img_shape = np.shape(img)
    img_list = np.reshape(img, (-1, *img_shape[-2:]))

    if mask is not None:
        img[mask] = 0
    coords = [[], []]
    if interactive:
        figure1 = plt.figure()
        ax = figure1.add_subplot(111)
        plt.imshow(i)
        plt.text(x=-50, y=-20, s="Click to add points. Click again to remove points. You must have 5 points")

        def add_point(event):
            ix, iy = event.xdata, event.ydata
            x_diff, y_diff = np.abs(np.subtract(coords[0], ix)), np.abs(np.subtract(coords[1], iy))
            truth = list(np.multiply(x_diff < 10, y_diff < 10))
            if not len(x_diff):
                coords[0].append(ix)
                coords[1].append(iy)
            elif not any(truth):
                coords[0].append(ix)
                coords[1].append(iy)
            else:
                truth = list(np.multiply(x_diff < 10, y_diff < 10)).index(True)
                coords[0].pop(truth)
                coords[1].pop(truth)
            print(coords)
            ax.clear()
            ax.imshow(img)
            ax.scatter(coords[0], coords[1], s=10, marker="o", color="crimson")
            figure1.canvas.draw()
        cid = figure1.canvas.mpl_connect('button_press_event', add_point)
        plt.show()
        print(coords[0])
    #  non-interactive, works better if there is an intense ring
    # TODO: Make method more robust with respect to obviously wrong points
    else:
        i_shape = np.shape(img)
        print(i_shape)
        flattened_array = img.flatten()
        indexes = sorted(range(len(flattened_array)), key=flattened_array.__getitem__)
        # take top 5000 points make sure exclude zero beam
        coords[0] = np.remainder(indexes[-num_points:], i_shape[0])  # x axis (column)
        coords[1] = np.floor_divide(indexes[-num_points:], i_shape[1])  # y axis (row)
    a = fit_ellipse(np.array(coords[0]), np.array(coords[1]))
    center = ellipse_center(a)  # (x,y)
    #  center = [center[1],center[0]] # array coordinates (y,x)
    lengths = ellipse_axis_length(a)
    angle = ellipse_angle_of_rotation(a)
    #  angle = (np.pi/2)-angle # transforming to array coordinates
    print("The center is:", center)
    print("The major and minor axis lengths are:", lengths)
    print("The angle of rotation is:", angle)
    if plot:
        ellipse = Ellipse((center[0], center[1]), lengths[0] * 2, lengths[1] * 2, angle=angle, fill=False)
        fig = plt.figure()
        axe = fig.add_subplot(111)
        axe.imshow(img)
        axe.add_patch(ellipse)
        plt.show()
    print(img_shape)
    return center, lengths, angle

