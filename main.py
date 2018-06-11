from feature_extractor import get_features
from xg_model import xgb_score

if __name__ == '__main__':
    train_features, test_features, train_flg, test_flg, feature_types = get_features(regenerate=True)

