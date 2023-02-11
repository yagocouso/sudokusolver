import cv2
import numpy as np
import time
from sklearn.cluster import KMeans
from keras.models import load_model


class SudokuSolver():

    # Create a 9x9 matrix
    __sudoku = np.full(shape=[9, 9], fill_value=None)

    # Crete a matrix of points in the image, reference to sudoku
    points = np.empty(shape=[9, 9, 2], dtype=int)

    # Create a numbers of sudoku
    __unique = set(range(1, 10))

    # Select font for the numbers in images
    _font = cv2.FONT_HERSHEY_SIMPLEX     

    _row = []
    _col = []

    def __init__(self, path = './sudoku.png', model = "./model.h5"):
        # Load image
        self.__image = cv2.imread(path)
        self.__biColor = self.__image.copy()

        # Load model
        self.__model = load_model(model)

        # Convert to grees scale
        biColor = cv2.cvtColor(self.__biColor, cv2.COLOR_BGR2GRAY)

        # Detectamos bordes con el método Cany
        canny = cv2.Canny(biColor, 50, 150)

        # Apply detect lines with Hough
        lines = cv2.HoughLines(canny, 1, np.pi/180, 200) 

        # Get cross between horizontal and vertical lines
        self.get_crosses(lines)

        # Get points from crosses
        self._row = self.get_points(self._row)
        self._col = self.get_points(self._col)

        # Insert numbers read in the image
        self.asign_image_number(biColor)

    @property
    def font(self):
        return self._font

    @font.setter
    def font(self, new_font):
        self._font = new_font

    def update_model(self, path):
        self.__model = load_model(path)

    def solver(self):
        """
        
        Returns
        -------
        Integer
            Next number
        Integer
            Row number.
        Integer
            Column number.

        """
        for row in range(9):
            for col in range(9):
                if self.__sudoku[row, col] != None: continue
                dataset = self.__unique - self.set_row(self.__sudoku, row) \
                    - self.set_columns(self.__sudoku, col) \
                    - self.set_square(self.__sudoku, row, col)
                if len(dataset) > 1: 
                    self.__sudoku[row, col] = None
                    continue
                self.__sudoku[row, col] = list(dataset)[0]
                return self.__sudoku[row, col], row, col
        return None, None, None
        

    def run(self):

        while True:
            # Show the image
            cv2.imshow('frame', self.__image)
            
            # Searching empty cells
            number, row, col = self.solver()

            # Checking if we can't solver more
            finish = not number and not row and not col
            
            # If it have finished or we have push the 'q' for quit of the aplication
            if finish or cv2.waitKey(1) == ord('q'):
                time.sleep(2 if finish else 0)
                break
            
            # Putting the next number text
            self.__image = cv2.putText(self.__image, str(number), tuple(self.points[row, col]), self._font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Wait a bit time for watch the image
            time.sleep(0.8)

        # Close windows
        cv2.destroyAllWindows()
        
    @staticmethod
    def set_square(array, row, col):
        """
        Calcule the numbers in the frame and return
        a list with that numbers
        
        Parameters
        ----------
        array : numpy array 9x9
            Sudoku array.
        row : integer
            Row number (started 0).
        col : integer
            Column number (started 0).

        Returns
        -------
        Set
            Numbers in the 3x3 frame of sudoku.

        """
        crow, ccol = (row // 3) * 3, (col // 3) * 3
        return set(array[crow:crow + 3, ccol:ccol + 3].reshape(array.shape[0],))

    @staticmethod
    def set_row(array, row):
        """
        

        Parameters
        ----------
        array : TYPE
            DESCRIPTION.
        row : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return set(array[row, :])

    @staticmethod
    def set_columns(array, col):
        """
        

        Parameters
        ----------
        array : TYPE
            DESCRIPTION.
        col : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return set(array[:, col])

    def get_points(self, points):
        """
        Get points in the sudoku template

        Parameters
        ----------
        points : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        values = np.array(points).astype(int)
        values = KMeans(n_clusters = 10, random_state=0).fit(values).cluster_centers_
        values = values[np.argsort(values[:, 0])].astype(int)
        values[1:, 0] = values[:-1, 1]
        return values[1:, :]

    def preprocesing(self, cell_img):
        """
        From matrix image, convert and extract the number inside
        in a number machine.

        Parameters
        ----------
        cell_img : Array
            Frma of image with a cell of sudoku

        Returns
        -------
        Float
            Number in the image

        """
        px = 28
        cell_img = cv2.resize(cell_img, (px, px), interpolation = cv2.INTER_AREA)
        inverted_image = cv2.bitwise_not(cell_img)
        inverted_image = inverted_image.reshape(1, px*px).astype('float32') / 255
        # inverted_image = inverted_image 
        return float(np.argmax(self.__model.predict(inverted_image)))
    

    def get_crosses(self, lines):
        """
        This fuction get a cross in the image

        Parameters
        ----------
        lines : Array
            Points that definy a line.

        Returns
        -------
        row : Array
            List .
        col : Array
            DESCRIPTION.

        """
        lines = lines.reshape(lines.shape[0], lines.shape[2])
        for r, theta in lines: 
            a, b = np.cos(theta), np.sin(theta) 
            x0, y0 = a*r, b*r
            if a < 1: self._col.append([y0 + 1000*(a), y0 - 1000*(a)])
            if b < 1: self._row.append([x0 + 1000*(-b), x0 - 1000*(-b)])


    def asign_image_number(self, biColor):
        row_list, col_list = [], []
        # Get points from lines
        elements = list(range(self._col.shape[0]))
        for i in elements:
            row_list += elements
            col_list += elements[i:] + elements[:i]
            
        cuts = np.concatenate((self._row[row_list], self._col[col_list]), axis=1)

        for i in range(len(cuts)):
            self.points[row_list[i], col_list[i]] = [cuts[i, 2] + 10, cuts[i, 1] - 10]
            number = self.preprocesing(biColor[cuts[i, 0]:cuts[i, 1], cuts[i, 2]:cuts[i, 3]])
            if not int(number): continue
            self.__sudoku[row_list[i], col_list[i]] = number
            cv2.putText(self.__image, str(int(self.__sudoku[row_list[i], col_list[i]])), tuple(self.points[row_list[i], col_list[i]]), self._font,1,(155, 68, 236), 2, cv2.LINE_AA)



    
    
        


    

