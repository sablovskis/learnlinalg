from math import sqrt


class Matrix():

    # Store the matrix as list of columns
    # TODO should I save lists and scalares correctly?
    def __init__(self, value):
        self.value = value
        self.pos = 0
    
    
    # What the computer sees
    def __repr__(self):
        return str(self.value)


    # What the user sees
    def __str__(self):
        return draw_matrix(self)


    # Make it posible to access the matrix as a list
    def __getitem__(self, position):
        return self.value[position]


    # Return the number of columns in the matrix
    def __len__(self):
        return len(self.value)


    # Addition using plus sign:
    def __add__(self, other):
        if self.value == 'Empty' or other.value == 'Empty':
            return Matrix('Empty')
        return add(self, other)


    # Subtraction using minus sign:
    def __sub__(self, other):
        if self.value == 'Empty' or other.value == 'Empty':
            return Matrix('Empty')
        return add(self, other, False)


    # Multiplication using star sign:
    def __mul__(self, other):
        if self.value == 'Empty' or other.value == 'Empty':
            return Matrix('Empty')        
        return multiply(self, other)


    # Divides matrix by a scalar
    def __truediv__(self, other):
        if self.value == 'Empty' or other.value == 'Empty':
            return Matrix('Empty')        
        return divide(self, other)


    # Return the dimensions of matrix as (columns, rows)
    def size(self):
        return size(self)


    # Returns True if matrix is square
    def square(self):
        if size(self)[0] == size(self)[1]:
            return True
        else:
            return False


    # Returns the size of the diagonal
    def diagonal_length(self):
        return min(size(self))


    # Get vector length
    def norm2(self):
        return norm2(self)


    # Transpose matrix
    def t(self):
        return transpose(self)

    # Invert matrix
    def i(self):
        size = self.size()
        if size[0] != size[1]:
            raise Exception('The matrix has to be square.')
        return gauss_jordan(self, eye(size[0]))[0]

    # Return indicated column as a vector
    def column(self, number):
        result = [self[number]]
        return Matrix(result)


    # Return indicated row as a vector
    def row(self, number):
        result = [[self[i][number] for i in range(self.size()[0])]]
        return Matrix(result)


    @property # Get a00 slice of matrix
    def a00(self):
        if self.pos == 0: # No matrix in position 0
            return Matrix('Empty')
        return select(self, [0, 0], [self.pos - 1, self.pos - 1], True)
    
    @a00.setter # Set a00 slice of matrix
    def a00(self, top_matrix=False):
        if self.pos == 0: # Do nothing in positon 0
            return self
        self.value = update(self, top_matrix, 'a00', self.pos)
        return self

    
    @property # Get a10 slice of matrix
    def a10(self):
        if self.pos == 0: # No matrix in position 0
            return Matrix('Empty')      
        return select(self, [0, self.pos], [self.pos - 1, self.pos], True)
    
    @a10.setter # Set a10 slice of matrix
    def a10(self, top_matrix=False):
        if self.pos == 0: # Do nothing in positon 0
            return self
        self.value = update(self, top_matrix, 'a10', self.pos)
        return self


    @property # Get a20 slice of matrix
    def a20(self):
        if self.pos == 0: # No matrix in position 0
            return Matrix('Empty')
        if self.pos == self.diagonal_length() - 1: # No matrix in position Max
            return Matrix('Empty')
        return select(self, [0, self.pos + 1], [0, None], True)

    @a20.setter # Set a20 slice of matrix
    def a20(self, top_matrix=False):
        if self.pos == 0: # Do nothing in positon 0
            return self
        if self.pos == self.diagonal_length() - 1: # Do nothing in position Max
            return self
        self.value = update(self, top_matrix, 'a20', self.pos)
        return self


    @property # Get a01 slice of matrix
    def a01(self):
        if self.pos == 0: # No matrix in position 0
            return Matrix('Empty')
        return select_debug(self, [self.pos, 0], [self.pos, self.pos - 1], True)
    
    @a01.setter # Set a01 slice of matrix
    def a01(self, top_matrix=False):
        if self.pos == 0: # Do nothing in positon 0
            return self            
        self.value = update(self, top_matrix, 'a01', self.pos)
        return self


    @property # Get 11 slice of matrix
    def a11(self):
        return select(self, [self.pos, self.pos], [self.pos, self.pos], True)  
    @a11.setter # Set 11 slice of matrix
    def a11(self, top_matrix=False):
        self.value = update(self, top_matrix, 'a11', self.pos)
        return self


    @property # Get a21 slice of matrix
    def a21(self):
        if self.pos == self.diagonal_length() - 1: # No matrix in position Max
            return Matrix('Empty')            
        return select(self, [self.pos, self.pos + 1], [self.pos, None], True)

    @a21.setter # Set a21 slice of matrix
    def a21(self, top_matrix=False):
        if self.pos == self.diagonal_length() - 1: # Do nothing in position Max
            return self            
        self.value = update(self, top_matrix, 'a21', self.pos)
        return self        


    @property # Get a02 slice of matrix
    def a02(self):
        if self.pos == 0: # No matrix in position 0
            return Matrix('Empty')
        if self.pos == self.diagonal_length() - 1 and self.size()[0] == self.diagonal_length(): # No matrix in position Max
            return Matrix('Empty')            
        return select(self, [self.pos + 1, 0], [None, self.pos - 1], True)

    @a02.setter # Set a02 slice of matrix
    def a02(self, top_matrix=False):
        if self.pos == 0: # Do nothing in positon 0
            return self
        if self.pos == self.diagonal_length() - 1 and self.size()[0] == self.diagonal_length(): # Do nothing in position Max
            return self                
        self.value = update(self, top_matrix, 'a02', self.pos)
        return self        


    @property # Get a12 slice of matrix
    def a12(self):
        if self.pos == self.diagonal_length() - 1 and self.size()[0] == self.diagonal_length(): # No matrix in position Max
            return Matrix('Empty')
        return select(self, [self.pos + 1, self.pos], [None, self.pos], True)

    @a12.setter # Set a12 slice of matrix
    def a12(self, top_matrix=False):
        if self.pos == self.diagonal_length() - 1 and self.size()[0] == self.diagonal_length(): # Do nothing in position Max
            return self            
        self.value = update(self, top_matrix, 'a12', self.pos)
        return self        


    @property # Get a22 slice of matrix
    def a22(self):
        if self.pos == self.diagonal_length() - 1: # No matrix in position Max
            return Matrix('Empty')            
        return select(self, [self.pos + 1, self.pos + 1], allow_out=True)  

    @a22.setter # Set a22 slice of matrix
    def a22(self, top_matrix=False):
        if self.pos == self.diagonal_length() - 1: # Do nothing in position Max
            return self            
        self.value = update(self, top_matrix, 'a22', self.pos)
        return self        


def draw_matrix(matrix):
    # Get column widths
    column_widths = []
    for column in matrix:
        max_characters = 0
        for component in column:
            characters = len(str(component))
            if characters > max_characters:
                max_characters = characters
        column_widths.append(max_characters)
    # Store matrix as text
    matrix_as_text = ''
    for row in range(matrix.size()[1]):
        row_string = ''
        for column, component in enumerate(matrix.row(row)[0]):
            component *= 1000000
            component = int(component)
            component /= 1000000
            characters = len(str(component))
            spacing = column_widths[column] - characters + 1
            row_string += ( ' ' * spacing) + str(component)
        matrix_as_text += '\n' + row_string
    matrix_as_text += '\n'
    return matrix_as_text


def copy_matrix(matrix):
    result = [[i for i in matrix[column]] for column in range(matrix.size()[0])]
    return Matrix(result)


# Extracts the upper triangular matrix:
def extract_upper(matrix):
    # TODO challenge? do it with list comprehension
    # Get the diagonal lenght in order know how many rows we need to iterate through
    matrix = copy_matrix(matrix)
    diag = matrix.diagonal_length()
    for column in range(diag):
        for row in range(column + 1, diag):
            matrix[column][row] = 0
    return matrix


# Extracts the lower triangular matrix:
def extract_lower(matrix):
    width, height = matrix.size()[0], matrix.size()[1]
    result = zeros(width, height)
    for column in range(width):
        for row in range(height):
            if row >= column:
                result[column][row] = matrix[column][row]
    return result


def extract_diagonal(matrix, square=True):
    if square and not matrix.square():
        raise Exception(f'The matrix is not square ({matrix.size()}).')
    diag = matrix.diagonal_length()
    result = zeros(diag)
    for i in range(diag):
        result[i][i] = matrix[i][i]
    return result


# Pastes a matrix on top of another matrix
def overlay(matrix, top_matrix, top_left):
    size = matrix.size()
    # Calculate bottom right
    bottom_right = [top_left[0] + top_matrix.size()[0], top_left[1] + top_matrix.size()[1]]
    # Check if top_left is not outside of the matrix
    if top_left[0] > size[0] or top_left[1] > size[1]:
        raise Exception(f'top_left is outside of the matrix ({top_left} and {size}).')
    # Check if bottom_right is not outside of the matrix
    if bottom_right[0] > size[0] or bottom_right[1] > size[1]:
        raise Exception(f'bottom_right is outside of the matrix ({bottom_right} and {size}).')
    # Do the things
    matrix = copy_matrix(matrix)
    for column in range(top_left[0], bottom_right[0]):
        for row in range(top_left[1], bottom_right[1]):
            matrix[column][row] = top_matrix[column - top_left[0]][row - top_left[1]]
    return matrix


# Overlays a specified partition in matrix in specified diagonal position
def update(matrix, top_matrix, partition, pos):
    partitions = { # This is the position of top left corner
        'a00': [ 0,       0         ],
        'a10': [ 0,       pos       ],
        'a20': [ 0,       pos + 1   ],
        'a01': [ pos,     0         ],
        'a11': [ pos,     pos       ],
        'a21': [ pos,     pos + 1   ],
        'a02': [ pos + 1, 0         ],
        'a12': [ pos + 1, pos       ],
        'a22': [ pos + 1, pos + 1   ]
    }
    result = overlay(matrix, top_matrix, partitions[partition])
    return result


# Connects two matrices into one
def join(matrix_1, matrix_2, join_vertically=False):
    matrix_1_width, matrix_1_height = matrix_1.size()[0], matrix_1.size()[1]
    matrix_2_width, matrix_2_height = matrix_2.size()[0], matrix_2.size()[1]
    if join_vertically:
        if matrix_1_width != matrix_2_width:
            raise Exception(f'The width of matrices does not match ({matrix_1_width}, {matrix_2_width}).')
        joined_matrix = []
        for column in range(matrix_1_width):
            joined_matrix.append(matrix_1[column] + matrix_2[column])
        return Matrix(joined_matrix)
    else:
        if matrix_1_height != matrix_2_height:
            raise Exception(f'The width of matrices does not match ({matrix_1_height}, {matrix_2_height}).')
        joined_matrix = []
        for column in matrix_1:
            joined_matrix.append(column)
        for column in matrix_2:
            joined_matrix.append(column)
        return Matrix(joined_matrix)


# Makes all diagonal components 1
def overlay_unit(matrix):
    diag = matrix.diagonal_length()
    matrix = copy_matrix(matrix)
    for pos in range(diag):
        matrix[pos][pos] = 1
    return matrix


# Cuts out a new matrix from the specified top left and bottom right corner
def select(matrix, top_left, bottom_right=None, allow_out=False):
    size = matrix.size()
    # Set bottom_right to maximum if not provided
    if not bottom_right or bottom_right == [None, None]:
        bottom_right = size
    # Set bottom x or y to maximum if only one is provided
    else:
        if bottom_right[0] == None:
            bottom_right[0] = size[0]
            bottom_right[1] += 1
        elif bottom_right[1] == None:
            bottom_right[1] = size[1]
            bottom_right[0] += 1
        else:
            bottom_right[0] += 1
            bottom_right[1] += 1
    # Check if the bottom_right is not above or to the left of top_left
    if top_left[0] > bottom_right[0] or top_left[1] > bottom_right[1]:
        raise Exception(f'top_left can not be to the right or lower than bottom_right ({top_left[0]} and {bottom_right[0]}).')
    # Check if top_left is not outside of the matrix
    if top_left[0] >= size[0] or top_left[1] >= size[1] or top_left[0] < 0 or top_left[1] < 0:
        if allow_out:
            return zeros(1, 1)
        raise Exception(f'top_left is outside of the matrix ({top_left} and {size}).')
    # Check if bottom_right is not outside of the matrix
    if bottom_right[0] > size[0] or bottom_right[1] > size[1]:
        raise Exception(f'Bottom_right is outside of the matrix ({bottom_right} and {size}).')
    # Check if the selection is not smalled than 1x1 matrix
    if top_left == bottom_right:
        if allow_out and bottom_right == [0, 0]: # A case for selecting the initial a00 submatrix
            return zeros(1, 1)
        raise Exception(f'Can not return smaller than 1x1 matrix ({top_left} and {top_left}).')        
    # Build the selected matrix
    result = []
    for column in range(top_left[0], bottom_right[0]):
        result.append( matrix[column][top_left[1] : bottom_right[1]] )
    return Matrix(result)


# Cuts out a new matrix from the specified top left and bottom right corner
def select_debug(matrix, top_left, bottom_right=None, allow_out=False):
    size = matrix.size()
    # Set bottom_right to maximum if not provided
    if not bottom_right or bottom_right == [None, None]:
        bottom_right = size
    # Set bottom x or y to maximum if only one is provided
    else:
        if bottom_right[0] == None:
            bottom_right[0] = size[0]
            bottom_right[1] += 1
        elif bottom_right[1] == None:
            bottom_right[1] = size[1]
            bottom_right[0] += 1
        else:
            bottom_right[0] += 1
            bottom_right[1] += 1
    # Check if the bottom_right is not above or to the left of top_left
    if top_left[0] > bottom_right[0] or top_left[1] > bottom_right[1]:
        raise Exception(f'top_left can not be to the right or lower than bottom_right ({top_left[0]} and {bottom_right[0]}).')
    # Check if top_left is not outside of the matrix
    if top_left[0] >= size[0] or top_left[1] >= size[1] or top_left[0] < 0 or top_left[1] < 0:
        if allow_out:
            return zeros(1, 1)
        raise Exception(f'top_left is outside of the matrix ({top_left} and {size}).')
    # Check if bottom_right is not outside of the matrix
    if bottom_right[0] > size[0] or bottom_right[1] > size[1]:
        raise Exception(f'Bottom_right is outside of the matrix ({bottom_right} and {size}).')
    # Check if the selection is not smalled than 1x1 matrix
    if top_left == bottom_right:
        if allow_out and bottom_right == [0, 0]: # A case for selecting the initial a00 submatrix
            return zeros(1, 1)
        raise Exception(f'Can not return smaller than 1x1 matrix ({top_left} and {top_left}).')        
    # Build the selected matrix
    result = []
    for column in range(top_left[0], bottom_right[0]):
        result.append( matrix[column][top_left[1] : bottom_right[1]] )
    return Matrix(result)


# Slices the matrix into four or nine partitions
def slice_matrix(matrix, position, nine_partitions=False):
    pos = position
    # For the case of nine_partitions:
    a00 = select(matrix, [0, 0], [pos - 1, pos - 1], True)
    a01 = select(matrix, [pos, 0], [pos, pos - 1], True)
    a02 = select(matrix, [pos + 1, 0], [None, pos - 1], True)
    a10 = select(matrix, [0, pos], [pos - 1, pos], True)
    a20 = select(matrix, [0, pos + 1], [0, None], True)
    # For all cases:
    a11 = select(matrix, [pos, pos], [pos, pos])
    a12 = select(matrix, [pos + 1, pos], [None, pos], True)         
    a21 = select(matrix, [pos, pos + 1], [pos, None], True)       
    a22 = select(matrix, [pos + 1, pos + 1], allow_out=True)
    if nine_partitions:
        return a00, a01, a02, a10, a11, a12, a20, a21, a22
    else:
        return a11, a12, a21, a22


def slice_vector(vector, position):
    pos = position
    a1 = select(vector, [0, pos], [0, pos])
    a2 = select(vector, [0, pos + 1], [0, None], True)
    return a1, a2


# Also known as 2-norm I think
def norm2(matrix):
    # Make sure this is a matrix with one column
    if matrix.size()[0] != 1:
        raise Exception(f'This is a wrong size for a vector ({matrix.size()}).')
    # Return the square root of the sum of all components
    length = sqrt( ( matrix.t() * matrix )[0][0] )
    return Matrix([[length]])


def size(matrix):
    # Get the number of columns and rows
    x = len(matrix)
    y = len(matrix[0])
    # Check if all the columns are of equal size
    for i in matrix:
        if len(i) != y:
            return False
    return x, y # Not a Matrix object (2x int)


def add(matrix_1, matrix_2, sign=True):
    # Check if they have the same number of columns and rows
    size_1, size_2 = matrix_1.size(), matrix_2.size()
    if size_1 != size_2:
        raise Exception(f'The matrices are of wrong size ({size_1} and {size_2}).')
    # Zip two matrices and iterate
    result_matrix = []
    # Iterate through columns
    # Matrices   are shortened to 'm'
    # Columns    are shortened to 'col'
    # Components are shortened to 'c'
    for col_m_1, col_m_2 in zip(matrix_1, matrix_2):
        result_column = []
        # Iterate through components
        for c_col_m_1, c_col_m_2 in zip(col_m_1, col_m_2):
            # Check if sign is positive or negative
            if sign:
                result_component = c_col_m_1 + c_col_m_2
            else:
                result_component = c_col_m_1 - c_col_m_2
            result_column.append(result_component)
        result_matrix.append(result_column)
    return Matrix(result_matrix)


def multiply(matrix_1, matrix_2):
    # Check if the sizes are correct for multiplication
    size_1, size_2 = matrix_1.size(), matrix_2.size()
    if size_1[0] != size_2[1]:
        if size_1 == (1, 1) and size_2[1] != 1:
            # Changes the order in case scalar is put
            # in front of vector multiplication
            matrix_1, matrix_2 = matrix_2, matrix_1
        else:
            raise Exception(f'The matrices are of wrong size ({size_1} and {size_2}).')
    # Create correct number of columns in new matrix
    result = [[] for i in range(matrix_2.size()[0])]
    # Iterate through resulting matrix
    for row in range(matrix_1.size()[1]):
        for column in range(matrix_2.size()[0]):
            # Calculate resulting component
            new_component = dot_product(
                matrix_1.row(row),
                matrix_2.column(column)
                )
            result[column].append(new_component)
    return Matrix(result)


def divide(matrix, scalar):
    # # Check if we want to do a fraction
    # if matrix.size() == (1, 1):
    #     matrix, scalar = scalar, matrix
    #     matrix = copy_matrix(matrix)
    #     for column in range(matrix.size()[0]):
    #         for row in range(matrix.size()[1]):
    #             matrix[column][row] = scalar[0][0] / matrix[column][row]
    #             return matrix
    # Check if scalar is one component in size
    if scalar.size() != (1, 1):
        raise Exception(f'The provided scalar is not 1x1 in size ({scalar.size()}).')
    # Do a simple division
    matrix = copy_matrix(matrix)
    for column in range(matrix.size()[0]):
        for row in range(matrix.size()[1]):
            matrix[column][row] /= scalar[0][0]
    return matrix


def fraction(matrix):
    matrix = copy_matrix(matrix)
    for column in range(matrix.size()[0]):
        for row in range(matrix.size()[1]):
            matrix[column][row] = 1 / matrix[column][row]
    return matrix


# Does the row pivoting for a matrix.
# Outputs pivoted matrix and a pivot vector.
def row_pivoting(matrix, p, pos):
    a11, a12, a21, a22 = slice_matrix(matrix, pos)
    if a11[0][0] == 0:
        a11_a21 = join(a11, a21, True)
        # Create a temporary permutation vector
        p_temp = Matrix([[i for i in range(a11_a21.size()[1])]])
        for i, component in enumerate(a11_a21[0]):
            if component != 0:
                # Reorder the temporary permutation vector
                p_temp[0][0], p_temp[0][i] = i, 0
                # Apply pivoting to lower part of matrix
                lower_part_of_matrix = select(matrix, [0, pos])
                lower_part_of_matrix = apply_pivoting(lower_part_of_matrix, p_temp)
                matrix = update(matrix, lower_part_of_matrix, 'a10', pos)
                # matrix = overlay(matrix, lower_part_of_matrix, [0, pos])
                # Apply pivoting to output p vector
                p_selection = select(p, [0, pos])
                p_selection = apply_pivoting(p_selection, p_temp)
                p = update(p, p_selection, 'a10', pos)
                # p = overlay(p, p_selection, [0, pos])
                # Refresh sliced matrix for later computation
                a11, a12, a21, a22 = slice_matrix(matrix, pos)
                return matrix, p
        raise Exception(f'Could not find a row with component to replace with.')
    return matrix, p


# Returns Lower and Upper matrices on top of another
# (if you want to have them separate, you have to extract them
# and also not forget to apply unit diagonal on Lower triangle)
# LU factorization with Partial Pivoting = Gaussian Elimination with Row Swapping
# LU = pA (permutation matrix)
def LU(matrix):
    if not matrix.square():
        raise Exception(f'The matrix is not square ({matrix.size()}).')
    matrix = copy_matrix(matrix)
    diag = matrix.diagonal_length()
    # Create a permutation vector
    p = Matrix([[i for i in range(diag)]])
    # March through diagonal
    for pos in range(diag - 1):
        # Do the row pivoting
        matrix, p = row_pivoting(matrix, p, pos)
        # Refresh the matrix
        a11, a12, a21, a22 = slice_matrix(matrix, pos)
        # Calculate and set l21
        l21 = a21 / a11
        # matrix = update(matrix, l21, 'a21', pos)
        matrix = overlay(matrix, l21, [pos, pos + 1])
        # Calculate and set a22
        a22 = a22 - l21 * a12
        # matrix = update(matrix, a22, 'a22', pos)
        matrix = overlay(matrix, a22, [pos + 1, pos + 1])
    return matrix, p


# Note that b can be a matrix of multiple right hand sides
# NOTE: bug if matrix has more rows than columns
def gauss_jordan(m, b):
    diag = m.diagonal_length()
    b_columns = b.size()[0]
    p = Matrix([[i for i in range(diag)]])
    m = join(m, b)
    # Iterate through the diagonal  
    for pos in range(diag):
        m, p = row_pivoting(m, p, pos)
        m.pos = pos  
        # Calculate and update values
        m.a01 = m.a01 / m.a11
        m.a21 = m.a21 / m.a11
        m.a02 = m.a02 - m.a01 * m.a12
        m.a22 = m.a22 - m.a21 * m.a12
        m.a01 = zeros(*m.a01.size())
        m.a21 = zeros(*m.a21.size())
    multiplication_matrix = extract_gauss_jordan_multiplications(m)
    m = multiplication_matrix * m
    b = Matrix(m[ - b_columns : ])
    m = Matrix(m[ : b_columns ])
    return m, b    


# Both matrices have to be square
# Section 8.2.5
def gauss_jordan_invert_old_alternative(m, b):
    diag = m.diagonal_length()
    p = Matrix([[i for i in range(diag)]])
    m = copy_matrix(m)
    b = copy_matrix(b)
    # Iterate through the diagonal
    for pos in range(diag):
        m, p = row_pivoting(m, p, pos)
        b, p = row_pivoting(b, p, pos)
        m.pos, b.pos = pos, pos
        # Calculate and update values
        m.a01 = m.a01 / m.a11
        m.a02 = m.a02 - m.a01 * m.a12
        m.a21 = m.a21 / m.a11
        m.a22 = m.a22 - m.a21 * m.a12
        b.a00 = b.a00 - m.a01 * b.a10
        b.a01 = m.a01 * Matrix([[-1]])
        b.a20 = b.a20 - m.a21 * b.a10
        b.a21 = m.a21 * Matrix([[-1]])
        m.a01 = zeros(*m.a01.size())
        m.a21 = zeros(*m.a21.size())
        m.a12 = m.a12 / m.a11
        b.a10 = b.a10 / m.a11
        b.a11 = fraction( m.a11 )
        m.a11 = Matrix([[1]]) # perform last
    return m, b


def extract_gauss_jordan_multiplications(matrix):
    matrix = extract_diagonal(matrix, False)
    for i in range(matrix.diagonal_length()):
        matrix[i][i] = 1 / matrix[i][i]
    return matrix


# Overrides the same matrix
# Challenge from 8.2.5
def gauss_jordan_invert(m):
    m = copy_matrix(m)
    p = Matrix([[i for i in range(m.diagonal_length())]])
    # Iterate through the diagonal
    for pos in range(m.diagonal_length()):
        m, p = row_pivoting(m, p, pos)
        m.pos = pos
        # Calculate and update values
        m.a00 = m.a00 - m.a01 / m.a11 * m.a10
        m.a02 = m.a02 - m.a01 / m.a11 * m.a12
        m.a01 = m.a01 / m.a11 * Matrix([[-1]]) # last in row
        m.a20 = m.a20 - m.a21 / m.a11 * m.a10
        m.a22 = m.a22 - m.a21 / m.a11 * m.a12
        m.a21 = m.a21 / m.a11 * Matrix([[-1]]) # last in row
        m.a10 = m.a10 / m.a11
        m.a12 = m.a12 / m.a11
        m.a11 = fraction( m.a11 ) # last in row
    return m    


# Takes a pivot vector and swaps rows of matrix
def apply_pivoting(matrix, p):
    pivot_matrix = [[1 if pos==component else 0 for pos in range(p.size()[1])] for component in p[0]]
    return Matrix(pivot_matrix) * matrix


# Solves lower triangular matrix (Ly=b for y)
def solve_lower(matrix, b):
    b = copy_matrix(b)
    if not matrix.square():
        raise Exception(f'The matrix is not square ({matrix.size()}).')
    matrix = copy_matrix(matrix)
    diag = matrix.diagonal_length()
    # March down
    for pos in range(diag - 1):
        # Calculate and set b2
        b1 = select(b, [0, pos], [0, pos])
        b2 = select(b, [0, pos + 1])
        l21 = select(matrix, [pos, pos + 1], [pos, None])
        b2 = b2 - b1 * l21
        b = overlay(b, b2, [0, pos + 1])
    return b


# Solves upper triangular matrix (Ux=b for x)
def solve_upper(matrix, b):
    b = copy_matrix(b)
    if not matrix.square():
        raise Exception(f'The matrix is not square ({matrix.size()}).')
    matrix = copy_matrix(matrix)
    diag = matrix.diagonal_length()
    # March up
    for pos in reversed(range(diag)):
        # Calculate and set b1
        b1, b2 = slice_vector(b, pos)
        u11 = select(matrix, [pos, pos], [pos, pos])
        u12 = select(matrix, [pos + 1, pos], [None, pos], True)
        b1 = ( b1 - u12 * b2 ) / u11
        b = overlay(b, b1, [0, pos])
    return b


def gaussian_elimination(matrix):
    matrix = copy_matrix(matrix)
    diag = matrix.diagonal_length()
    for pos in range(diag - 1):
        # Slice the matrix
        a11, a12, a21, a22 = slice_matrix(matrix, pos)
        # Calculate and save coefficients
        l21 = a21 / a11
        matrix = overlay(matrix, l21, [pos, pos + 1])
        # Calculate and save a22 matrix
        a22 = a22 - l21 * a12
        matrix = overlay(matrix, a22, [pos + 1, pos + 1])
    return matrix


def forward_substitution(transformation_matrix, vector):
    if not transformation_matrix.square():
        raise Exception('Transformation matrix is not square.')    
    if transformation_matrix.size()[0] != vector.size()[1]:
        raise Exception('The width of matrix does not match the height of vector.')
    vector = copy_matrix(vector)
    # Do the transformations
    for pos in range(vector.size()[1] - 1):
        b1, b2 = slice_vector(vector, pos)
        a11, a12, a21, a22 = slice_matrix(transformation_matrix, pos)
        b2 = b2 - b1 * a21
        vector = overlay(vector, b2, [0, pos + 1])
    return vector


def axpy(scalar_a, vector_x, vector_y):
    # TODO Axpy operation is intended for vectors and scalars but
    # TODO now it works with matrices as well. Should I make it not
    # TODO work with matrices?
    return (scalar_a * vector_x) + vector_y


def dot_product(vector_1, vector_2):
    # Check if both are vectors and of same size
    size_1, size_2 = vector_1.size(), vector_2.size()
    if size_1 != size_2:
        raise Exception(f'The vectors are of wrong size ({size_1} and {size_2}).')
    # Run through both vectors
    result = 0
    for c_1, c_2 in zip(vector_1[0], vector_2[0]):
        # Add the result of all multiplications
        result += c_1 * c_2
    return result # Not a Matrix object (float)


def transpose(matrix):
    # Create correct number of columns in new matrix
    result = [[] for i in range(matrix.size()[1])]
    # Fill the new matrix with flipped values
    # TODO check if I am referencing the resulting or origin
    for row in range(matrix.size()[1]):
        for column in range(matrix.size()[0]):
            result[row].append(matrix[column][row])
    return Matrix(result)


# Generate the zero matrix
def zeros(columns, rows=None):
    if not rows:
        rows = columns
    result = [[0 for i in range(rows)] for i in range(columns)]
    return Matrix(result)


# Generates the identity matrix
def eye(size):
    # Generate zero matrix
    result = zeros(size, size)
    # Fill the diagonal with ones
    for column in range(size):
        result[column][column] = 1
    return result


# Generates a diagonal matrix from a vector
def diag(vector):
    # Check if the size is correct
    if vector.size()[0] != 1:
        raise Exception(f'Incorrect size provided ({vector.size()})')
    # Generate zero matrix
    size = vector.size()[1]
    result = zeros(size, size)
    # Fill the diagonal with values
    for i in range(size):
        result[i][i] = vector[0][i]
    return result