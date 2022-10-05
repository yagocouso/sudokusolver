# -*- coding: utf-8 -*-
"""
Created on Sun May  1 09:02:25 2022

@author: Yago
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
import time
from keras.models import load_model

sudoku = np.full(shape=[9, 9], fill_value=None)
row, col, row_list, col_list = [], [], [], []
unique = set(range(1, 10))
font = cv2.FONT_HERSHEY_SIMPLEX

# Load model
model = load_model("./model.h5")

# sudoku = np.array(
#     [[None, None, None, None, None, None, None, None, None],
#     [None, None, None, None, None, None, None, None, None],
#     [None, None, None, None, None, None, None, None, None],
#     [None, None, None, None, None, None, None, None, None],
#     [None, None, None, None, None, None, None, None, None],
#     [None, None, None, None, None, None, None, None, None],
#     [None, None, None, None, None, None, None, None, None],
#     [None, None, None, None, None, None, None, None, None],
#     [None, None, None, None, None, None, None, None, None]]
# )

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

def preprocesing(number_img):
    number_img = cv2.resize(number_img, (28, 28), interpolation = cv2.INTER_AREA)
    inverted_image = cv2.bitwise_not(number_img)
    inverted_image = inverted_image.reshape(1, 28*28).astype('float32')
    inverted_image = inverted_image / 255
    return float(np.argmax(model.predict(inverted_image)))

def get_crosses(lines):
    global row, col
    lines = lines.reshape(lines.shape[0], lines.shape[2])
    for r, theta in lines: 
        a, b = np.cos(theta), np.sin(theta) 
        x0, y0 = a*r, b*r
        if a < 1: col.append([y0 + 1000*(a), y0 - 1000*(a)])
        if b < 1: row.append([x0 + 1000*(-b), x0 - 1000*(-b)])
    return row, col






if __name__ == "__main__":
    # Load picture
    orig = cv2.imread('sudoku.png')
    
    # Convert to grees scale
    img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    
    # Detectamos bordes con el método Cany
    canny = cv2.Canny(img, 50, 150)
    
    # Apply detect lines with Hough
    lines = cv2.HoughLines(canny, 1, np.pi/180, 200) 
    
    # Get points from lines
    row, col = get_crosses(lines)
    row = get_points(row)
    col = get_points(col)
    
    elements = list(range(col.shape[0]))
    for i in elements:
        row_list += elements
        col_list += elements[i:] + elements[:i]
        
    cuts = np.concatenate((row[row_list], col[col_list]), axis=1)
    
    for i in range(len(cuts)):
        points[row_list[i], col_list[i]] = [cuts[i, 2] + 10, cuts[i, 1] - 10]
        number = preprocesing(img[cuts[i, 0]:cuts[i, 1], cuts[i, 2]:cuts[i, 3]])
        if not int(number): continue
        sudoku[row_list[i], col_list[i]] = number
        cv2.putText(orig, str(int(sudoku[row_list[i], col_list[i]])), tuple(points[row_list[i], col_list[i]]), font,1,(0, 0, 255), 2, cv2.LINE_AA)
        
    
    while True:
        number, row, col = solver()
        #Checking if we can't solver more
        finish = not number and not row and not col
        
        if finish or cv2.waitKey(1) == ord('q'):
            time.sleep(10 if finish else 0)
            break
        
        cv2.putText(orig, str(number), tuple(points[row, col]), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', orig)
        time.sleep(1)
        
    # When everything done, close windows
    cv2.destroyAllWindows()
    

