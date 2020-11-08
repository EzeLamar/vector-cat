#!/usr/bin/env python

import cv2
import itertools
import numpy
import os
import sys


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = os.path.abspath(sys.argv[1])
        output_path = os.path.abspath(".")
        
        # encode names
        namespath = os.path.join(path, 'names.txt')
        if os.path.isfile(namespath):
            with open(namespath) as namesfile:
                names = dict(enumerate(namesfile.readlines()))
                classes = list(map(int, sys.argv[2].split(','))) if len(sys.argv) > 2 else names.keys()
                
                titles = []
                for idx in classes:
                    titles += [names[idx].strip() + '_x', names[idx].strip() + '_y'] if idx in names else []
                titles += ['Image']
                
        output = []
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            
            if filename.endswith('.mta'):
                imagepath = os.path.join(path, filename.split('_meta')[0] + '.png')
                if os.path.isfile(imagepath):
                    with open(filepath) as infile:
                        # encode keypoints
                        keypoints = dict()
                        indicePunto = 0
                        for line in infile.readlines():
                            line = line.split(';')
                            if len(line) > 2:
                                keypoints[indicePunto] = (str(float(line[1].replace(',','.'))*96), str(96 - float(line[2].replace(',','.'))*96))
                                indicePunto += 1
                                
                        current = []
                        for idx in classes:
                            current += [','.join(keypoints[idx])] if idx in keypoints else [',']
                            
                        # encode image
                        image = cv2.cvtColor(cv2.imread(imagepath), cv2.COLOR_BGR2GRAY)
                        pixels = numpy.reshape(numpy.array(image, dtype = numpy.uint8), (1, -1))[0]
                        current += [' '.join(map(str, pixels))]
                        
                        output += [current]
                        
        with open(os.path.join(output_path, 'encoded.csv'), 'w') as outfile:
            outfile.write(','.join(titles) + '\n')
            outfile.write('\n'.join(list(map(lambda l: ','.join(l), output))))
    print("svc generated :)")
else:
    print("Error")
            