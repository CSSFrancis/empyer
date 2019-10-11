import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig, inv
from matplotlib.patches import Ellipse
from empyer.misc.cartesain_to_polar import convert


def solve_ellipse(img, interactive=False, num_points=500, plot=False, suspected_radius=None):
    """Takes a 2-d array and allows you to solve for the equivalent ellipse. Everything is done in array coord.

    Fitzgibbon, A. W., Fisher, R. B., Hill, F., & Eh, E. (1999). Direct Least Squres Fitting of Ellipses.
        IEEE Transactions on Pattern Analysis and Machine Intelligence, 21(5), 476–480.
    http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html

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
        x = x[:, np.newaxis]
        y = y[:, np.newaxis]
        D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))   # Design matrix [x^2, xy, y^2, x,y,1]
        S = np.dot(D.T, D)  # Scatter Matrix
        C = np.zeros([6, 6])
        C[0, 2] = C[2, 0] = 2
        C[1, 1] = -1
        E, V = eig(np.dot(inv(S), C))  # eigen decomposition to solve constrained minimization problem
        n = np.argmax(np.abs(E))   # maximum eigenvalue solution
        a = V[:, n]
        print("a is:", a)
        return a

    def ellipse_center(ellipse_parameters):
        a, b, c, d, e, f = ellipse_parameters
        denom = b**2 - (4 * a * c)
        x0 = (2 * c * d - b * e) / denom
        y0 = (2 * a * e - b * d) / denom
        return np.array([x0, y0])

    def ellipse_axis_length(ellipse_parameters):
        a, b, c, d, e, f = ellipse_parameters
        denom = b**2 - (4 * a * c)
        num1 = np.sqrt(2 * (a * e ** 2 + c * d ** 2 - b * d * e + (b ** 2 - 4 * a * c) * f) *
                       ((a+c) - np.sqrt((a - c) ** 2 + b ** 2)))
        num2 = np.sqrt(2 * (a * e ** 2 + c * d ** 2 - b * d * e + (b ** 2 - 4 * a * c) * f) *
                       ((a+c) + np.sqrt((a - c) ** 2 + b ** 2)))
        axis1 = abs(num1/denom)
        axis2 = abs(num2/denom)

        return np.sort([axis1, axis2])[::-1]

    def ellipse_angle_of_rotation(ellipse_parameters):
        a, b, c, d, e, f = ellipse_parameters
        b, d, e = b/2, d/2, e/3
        if b == 0:
            if a > c:
                return 0
            else:
                return np.pi / 2
        else:
            if a < c:
                ang = .5 * invcot((a-c)/(2*b))
                if (a<0) == (b<0): # same sign
                    ang = ang+np.pi/2
                return ang
            else:
                ang =np.pi/2 + .5 * invcot((a-c)/(2*b))
                if (a < 0) != (b < 0):
                    ang=ang-np.pi/2
                return ang


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

    #  non-interactive, works better if there is an intense ring
    # TODO: Make method more robust with respect to obviously wrong points
    else:
        coords = get_max_positions(img, num_points=num_points, radius=suspected_radius)
    a = fit_ellipse(np.array(coords[0]), np.array(coords[1]))
    center = ellipse_center(a)  # (x,y)
    lengths = ellipse_axis_length(a)
    angle = ellipse_angle_of_rotation(a)
    print("The center is:", center)
    print("The major and minor axis lengths are:", lengths)
    print("The angle of rotation is:", angle)
    if plot:
        print("plotting")
        plt.scatter(coords[0], coords[1])
        ellipse = Ellipse((center[1], center[0]), lengths[0] * 2, lengths[1] * 2, angle=angle, fill=False)
        fig = plt.figure()
        axe = fig.add_subplot(111)
        axe.imshow(img)
        axe.add_patch(ellipse)
        plt.show()
    return center, lengths, angle


def advanced_solve_ellipse(img, center, lengths, angle, phase_width, radius, num_points=500):
    """ This is a method in development. Better at optimizing angular correlations

    This method is in development. Due to the fact that an ellipse will have maximum 2-fold symmetry when the center
    is correctly determined and maximum 2*n-fold symmetry when the major/ minor axes as well as the angle of rotation
    is correct. This algorithm optimizes for these quantities.

    Parameters:
    -----------
    img: Array-like
        2-d Array of some image
    range: list
        The lower and upper limits to look at. Usually should just look at the first ring.
    num_points: int
        The number of points to look at to determine the characteristic ellipse.

    Return:
    -------------
    center: list
        The x and y coordinates of the center
    lengths: list
        The major and minor axes
    angle: float
        The angle of rotation for the ellipse
    """

    # Brute force testing method...

    x = np.linspace(-5,5,20)
    y = np.linspace(-5, 5, 20)

    pol =[[convert(img, angle=None, lengths=None, center=np.add(center[x1,y1]), phase_width=phase_width, radius=radius)
           for x1 in x]for y1 in y]
    


def invcot(val):
    return (np.pi/2) - np.arctan(val)


def get_max_positions(image, num_points=None, radius=None):
    i_shape = np.shape(image)
    flattened_array = image.flatten()
    indexes = np.argsort(flattened_array)

    if isinstance(flattened_array, np.ma.masked_array):
        indexes = indexes[flattened_array.mask[indexes] == False]
    if radius is not None:
        center = [np.floor_divide(np.mean(indexes[-num_points:]), i_shape[1]),
                  np.remainder(np.mean(indexes[-num_points:]), i_shape[1])]
        print(center)
    # take top 5000 points make sure exclude zero beam
    cords = [np.floor_divide(indexes[-num_points:], i_shape[1]),
             np.remainder(indexes[-num_points:], i_shape[1])]  # [x axis (row),y axis (col)]
    return cords