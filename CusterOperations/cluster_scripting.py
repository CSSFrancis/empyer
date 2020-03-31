#!/usr/bin/python
import os
import glob
import empyer
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
import time



# This program is designed to take a diffraction signal, convert the signal to polar coordinates and then output
# the angular correlation and power spectrum.

"""
OPTIONS:
-d = cwd :directory
"""


def correlation(signal_file, args):
    t0=time.time()
    ds = empyer.load(signal_file, signal_type='diffraction_signal')
    if args.mask_below:
        ds.mask_below(args.mask_below)
    if args.border_mask:
        ds.mask_border(pixels=args.border_mask)
    if args.box_mask:
        l = np.array(args.box_mask,dtype=float)
        ds.masig[l[0]:l[1], l[2]:l[3]] = True
    if args.circle_mask:
        l =np.array(args.circle_mask,dtype=float)
        ds.mask_circle(l[0],l[1],l[2])

    file_name = os.path.splitext(signal_file)[0]
    ps = ds.calculate_polar_spectrum()
    ps.save(filename=file_name+'_polar.hdf5', overwrite=True)
    acs = ps.to_correlation()
    acs.save(filename=file_name + '_angular.hdf5', overwrite=True)
    pow_s = acs.get_power_spectrum()
    pow_s.save(filename=file_name+'_angularPower.hdf5', overwrite=True)
    tf=time.time()
    print(file_name+ " Finished converting in " + str(tf-t0) + " seconds")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--directory",
                        type=str,
                        help="input directory")
    parser.add_argument("-mb",
                        "--mask_below",
                        type=float,
                        help="Mask any values below this value")
    parser.add_argument("-bm",
                        "--border_mask",
                        type=float,
                        help="Mask this many pixels on the boarder of the image")
    parser.add_argument("-b",
                        "--box_mask",
                        nargs="+",
                        help="A list of the coordinates for a box mask applied to the image")
    parser.add_argument("-c",
                        "--circle_mask",
                        nargs="+",
                        help="The coordinates for a circular mask ")
    args = parser.parse_args()

    if not args.directory:
        args.directory = os.getcwd()

    if os.path.isdir(args.directory):
        files = glob.glob(args.directory+"/*.hdf5")
    else:
        files = [args.directory]

    if os.path.isdir(args.directory):
        files = glob.glob(args.directory+"/*.hdf5")
    else:
        files = [args.directory]
    print("The parameters are:" + str(args))
    print("Allocating", 12, " Processers")
    start_stop = []
    p = Pool(processes=12, maxtasksperchild=1)  # reallocating memory after every process finishes.
    for f in files:
        correlation(f, args)
    tasks = [p.apply_async(correlation, (f, args)) for f in files]
    for t in tasks:
        t.get()

    p.close()
    p.join()