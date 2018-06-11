from feature_extractor import get_features
from xg_model import xgb_score

if __name__ == '__main__':
    features, feature_type, flg = get_features(regenerate=True)

