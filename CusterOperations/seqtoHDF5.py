#!/usr/bin/python
import struct
import argparse
import numpy as np
import time
from libtiff import TIFF
import os


def loadMRCfile(filepath):
    with open(filepath, mode='rb') as file:
        # getting the size of the mrc image
        file.seek(0)
        read_bytes = file.read(8)
        frame_width = struct.unpack('<i', read_bytes[0:4])[0]
        frame_height = struct.unpack('<i', read_bytes[4:8])[0]
        file.seek(256 * 4)
        #reading the images
        dataset = file.read(frame_width * frame_height * 4)
        image = np.reshape(np.frombuffer(dataset, dtype=np.float32),(frame_height,frame_width))
        return image


def loadHeader(fileName, darkref):
    print('Reading file ' + fileName)
    with open(fileName, mode='rb') as file:  # b is important -> binary
        file.seek(548)
        read_bytes = file.read(20)
        frame_width = struct.unpack('<L', read_bytes[0:4])
        frame_height = struct.unpack('<L', read_bytes[4:8])
        print('Each frame is ' + str(frame_width[0]) + ' by ' + str(frame_width[0]) + ' px.')

        file.seek(572)
        read_bytes = file.read(4)
        num_frames = struct.unpack('<i', read_bytes)
        print('Total ' + str(num_frames[0]) + ' frames collected.')

        file.seek(584)
        read_bytes = file.read(8)
        frame_rate = struct.unpack('<d', read_bytes)
        print('Image acquired at ' + str(frame_rate[0]) + ' frames per second.')

        file.seek(580)
        read_bytes = file.read(4)
        true_imagesize = struct.unpack('<L', read_bytes[0:4])

        if frame_width[0] != darkref.shape[0] or frame_height[0] != darkref.shape[1]:
            print('Norpix frame size (' + str(frame_height[0]) + ',' + str(
                frame_width[0]) + ') disagree with reference size' + str(darkref.shape))

        return num_frames[0], true_imagesize[0]


def saveFile(directory, filename, darkref, gainref, numframes, true_imagesize, savename):
    try:
        os.mkdir(directory + savename)
    except OSError as e:
        print("Directory already exists")
    with open(directory+filename, mode='rb') as file:
        print(numframes)
        s = time.time()
        for iframe in range(numframes):
            filename = "Image" + format(iframe + 1, '05') + '.tiff'
            file.seek(8192 + iframe * true_imagesize)
            read_bytes = file.read(darkref.shape[0] * darkref.shape[1] * 2)
            # loading from buffer
            frame_raw = np.reshape(np.frombuffer(read_bytes, dtype=np.uint16), darkref.shape)
            frame = (frame_raw - darkref) * gainref
            # change to 32 bit images because that is how the gain and reference are collected.
            frame = np.array(frame, dtype=np.int32)
            # saving as 32 bit tif
            tiff = TIFF.open(directory+savename+"/" + filename, mode='w')
            tiff.write_image(frame)
            tiff.close()
        st = time.time()
        print("Time for the conversion of one Image is: ", (st-s)/numframes, "seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--directory",
                        type=str,
                        help="input directory")
    parser.add_argument("-f",
                        "--filename",
                        type=str,
                        help="base filename")
    parser.add_argument("-dr",
                        "--darkreference",
                        type=str,
                        help="dark reference")
    parser.add_argument("-g",
                        "--gainreference",
                        type=str,
                        help="gain reference")
    parser.add_argument("-s",
                        "--savename",
                        type=str,
                        help="saving directory name")
    args = parser.parse_args()

    if not args.directory:
        args.directory = os.getcwd()
    start = time.time()
    # loading the dark and gain references
    darkref = loadMRCfile(args.directory + args.darkreference)
    gainref = loadMRCfile(args.directory + args.gainreference)

    if darkref.shape != gainref.shape:
        print('Dark and gain reference shape disagree!')
    numframes, true_imagesize = loadHeader(args.directory+args.filename, darkref)
    # loading the images from the .seq file and then saving them as tif files
    saveFile(directory=args.directory,
             filename=args.filename,
             darkref=darkref,
             gainref=gainref,
             numframes=numframes,
             true_imagesize=true_imagesize,
             savename=args.savename)
    end = time.time()
    print('Total time elapsed: ' + str(end - start))