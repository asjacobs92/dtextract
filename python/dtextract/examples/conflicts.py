# Main code

from .runCompare import *
from ..data.consts import *

if __name__ == "__main__":
    runCompare(CONFLICTS_PATH, CONFLICTS_HAS_HEADER, CONFLICTS_DATA_TYPES,
               CONFLICTS_IS_CLASSIFY, CONFLICTS_N_DATA_MATRIX_COLS, CONFLICTS_OUTPUT)
