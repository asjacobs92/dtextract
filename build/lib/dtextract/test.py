# Main code
from sklearn.ensemble import RandomForestClassifier

from data.consts import *
from base import extract

if __name__ == "__main__":
    dt = extract(RandomForestClassifier,
                 CONFLICTS_PATH,
                 CONFLICTS_DATA_TYPES,
                 hasHeaders=CONFLICTS_HAS_HEADER,
                 headers=CONFLICTS_HEADERS)

    print(str(dt))
