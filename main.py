from feature_extractor import get_features
from xg_model import xgb_score

if __name__ == '__main__':
    train, test, feature_types = get_features(regenerate=True)
    train_features = train[0]
    train_flg = train[1]
    test_features = test[0]
    test_flg = test[1]
    

