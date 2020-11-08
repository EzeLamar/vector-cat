#!/usr/bin/env python

import cv2
import numpy
import os
import sys


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = os.path.abspath(sys.argv[1])
        path = filename.split('.')[0]
        
        if not os.path.isdir(path):
            os.mkdir(path)
            
        with open(filename) as infile:
            for idx, line in enumerate(infile):
                columns = line.split(',')
                if idx == 0:    # info row
                    # recover names
                    names = list(map(lambda x: x[:-2], columns[:-1:2]));
                    with open(os.path.join(path, 'names.txt'), 'w') as outfile:
                        for name in names:
                            outfile.write('{}\n'.format(name))
                            
                else:           # data rows
                    # recover keypoints
                    keypoints = list(map(lambda x: float(x) / 96 if x else None, columns[:-1]))
                    keypoints = list(zip(*[keypoints[i::2] for i in range(2)]))
                    with open(os.path.join(path, 'image{}.txt'.format(idx)), 'w') as outfile:
                        for id, (x, y) in enumerate(keypoints):
                            if x is not None and y is not None:
                                outfile.write('{} {} {} 0.01 0.01\n'.format(id, x, y))
                                
                    # recover image
                    pixels = columns[-1].split()
                    pixels = numpy.array(list(map(int, pixels)), dtype = numpy.uint8)
                    image = numpy.reshape(pixels, (-1, 96))
                    
                    cv2.imwrite(os.path.join(path, 'image{}.png'.format(idx)), image)
                    