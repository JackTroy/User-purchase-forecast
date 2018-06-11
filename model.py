# model should be here

from data_handler import get_data
from feature_extractor import get_feature

def model():
    pass

if __name__ == '__main__':
    agg, log, flg = get_data()
    features, feature_type = get_feature(agg, log, flg)
