
# threshold 
BINARY_THRESHOLD = 150  # from 80 to 150
BINARY_THRESHOLD_EASY_1 = 166  # idx = 0
BINARY_THRESHOLD_HARD_2 = 80   # idx = 4
BINARY_THRESHOLD_MEDIUM_1 = 75 # idx = 6
BINARY_THRESHOLD_MEDIUM_3 = 75 # idx = 8

CANNY_THRESHOLD_1 = 100 
CANNY_THRESHOLD_2 = 200 

# kernel size
DILATED_KERNEL_SIZE = (5, 5) # don't change this value, change its iterations (DILATED_ITERATIONS) value
ERODED_KERNEL_SIZE = (5, 5)
MORPHOLOGY_KERNEL_SIZE = (10, 10) # from 10 to 18

# iterations
DILATED_ITERATIONS = 4 
ERODED_ITERATIONS = 1

# place
MAX_ROW_LIMIT = 3;
MAX_COLUMN_LIMIT = 3;

# color
COLOR_WHITE_1 = 255
COLOR_BLACK_1 = 0
COLOR_WHITE_3 = (255, 255, 255)
COLOR_BLACK_3 = (0, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)
COLOR_BLUE = (0, 0, 255)

COLOR_FROM_DEFAULT = COLOR_BLACK_1
COLOR_TO_DEFAULT = COLOR_GREEN

# max size fim
MAX_SIZE_DIM = 300