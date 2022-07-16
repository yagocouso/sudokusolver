# -*- coding: utf-8 -*-
"""
Created on Sun May  1 09:02:25 2022

@author: Yago
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time

sudoku = np.zeros(shape=[9, 9])
row, col = [], []
row_list, col_list = [], []
unique = set(range(1, 10))
font = cv2.FONT_HERSHEY_SIMPLEX

# TODO delete
sudoku = np.array(
    [[5., 3., None, None, 7., None, None, None, None],
    [6., None, None, 1., 9., 5., None, None, None],
    [None, 9, 8., None, None, None, None, 6., None],
    [8., None, None, None, 6., None, None, None, 3.],
    [4., None, None, 8., None, 3., None, None, 1.],
    [7., None, None, None, 2., None, None, None, 6.],
    [None, 6, None, None, None, None, 2., 8., None],
    [None, None, None, 4., 1., 9., None, None, None],
    [None, None, None, None, 8., None, None, 7., 9.]]
)

points = np.empty(shape=[9, 9, 2], dtype=int)



def get_points(points):
    values = np.array(points).astype(int)
    values = KMeans(n_clusters = 10, random_state=0).fit(values).cluster_centers_
    values = values[np.argsort(values[:, 0])].astype(int)
    values[1:, 0] = values[:-1, 1]
    return values[1:, :]

def set_square(array, row, col):
    crow, ccol = (row // 3) * 3, (col // 3) * 3
    return set(array[crow:crow + 3, ccol:ccol + 3].reshape(array.shape[0],))

def set_row(array, row):
    return set(array[row, :])

def set_col(array, col):
    return set(array[:, col])

def solver():
    global sudoku, unique
    for row in range(9):
        for col in range(9):
            if sudoku[row, col] != None: continue
            dataset = unique - set_row(sudoku, row) - set_col(sudoku, col) - set_square(sudoku, row, col)
            if len(dataset) > 1: 
                sudoku[row, col] = None
                continue
            sudoku[row, col] = list(dataset)[0]
            return sudoku[row, col], row, col
    return None, None, None

img = cv2.imread('sudoku.png')


# Load picture
img = cv2.imread('sudoku.png')

# Convert to grees scale
cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detectamos bordes con el método Cany
canny = cv2.Canny(img, 50, 150)

# Apply detect lines with Hough
lines = cv2.HoughLines(canny,1,np.pi/180, 200) 
lines = lines.reshape(lines.shape[0], lines.shape[2])

for r, theta in lines: 
    a, b = np.cos(theta), np.sin(theta) 
    x0, y0 = a*r, b*r
    if a < 1: col.append([y0 + 1000*(a), y0 - 1000*(a)])
    if b < 1: row.append([x0 + 1000*(-b), x0 - 1000*(-b)])


row = get_points(row)
col = get_points(col)

elements = list(range(col.shape[0]))
for i in elements:
    row_list += elements
    col_list += elements[i:] + elements[:i]
    

cuts = np.concatenate((row[row_list], col[col_list]), axis=1)

for i in range(len(cuts)):
    points[row_list[i], col_list[i]] = [cuts[i, 0], cuts[i, 3]]
    
    if sudoku[row_list[i], col_list[i]] == None: continue
    
    cv2.putText(img, str(int(sudoku[row_list[i], col_list[i]])), tuple(points[row_list[i], col_list[i]]), font,1,(0, 0, 255), 2, cv2.LINE_AA)
    # sudoku[lista_y[i], lista_x[i]] = i
    # cv2.imshow('image',canny[cuts[i, 0]:cuts[i, 1], cuts[i, 2]:cuts[i, 3]])
    # cv2.waitKey(0)
    # cv2.rectangle(img,(cuts[i, 0],cuts[i, 2]),(cuts[i, 1],cuts[i, 3]),(0,255,0),1)

# cv2.imshow('image',img)q
# cv2.waitKey(0)

while True:
    # Capture frame-by-frame
    frame = img.copy()

    number, row, col = solver()
    
    cv2.putText(frame, str(number), tuple(points[row, col]), font,1,(0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, str(number), tuple(points[row, col]), font,1,(0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    time.sleep(1)
    if cv2.waitKey(1) == ord('q'):
        break
    

# When everything done, release the capture
cv2.destroyAllWindows()


