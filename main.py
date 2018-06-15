from feature_extractor import get_features
from xg_model import xgb_score
from xg_model import xgb_model
from xg_model import save_result

if __name__ == '__main__':
    train, test, feature_types = get_features(regenerate=False)
    print(train[0].head())
    # get model score
    # the default cv is 5
    xgb_score(train)

    # get model predict
    #predict = xgb_model(train[0], train[1], test[0])

    # save model predict to prediction/
    # return is the csv content. type is Dataframe
    # the default csv name is test_result
    result = save_result(train, test)
    print(result.head())
