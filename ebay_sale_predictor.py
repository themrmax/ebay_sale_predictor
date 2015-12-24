import requests
import re
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import argparse
import pickle
import os



def parse_search_results(t):
    ad_title = re.findall("alt='([^']*)", t) #don't parse html with regex because tony the pony
    n_bids = re.findall("(\d+) bid", t)
    price = re.findall("\$</b>(\d+)", t)

    if not (len(ad_title) == len(n_bids) and len(n_bids) == len(price)):
        raise Exception("Parsing failed, length mismatch between ad text, bids and price vectors")
    return pd.DataFrame({'ad_title':ad_title, 'n_bids':n_bids, 'price':price})


def get_raw_search_results(search_term, page_number):
    url = 'http://www.ebay.com.au/sch/i.html?_fcid=15&LH_Auction=1&LH_Complete=1&_clu=2&gbr=1&_from=R40&_sacat=0&_nkw={}&_ipg=200&rt=nc&_pgn='.format(search_term)
    return requests.get(url+str(page_number)).text


def munge_data(raw_data):
    munged = (raw_data.assign(ad_title = lambda x: x.ad_title.str.lower().str.replace('[^a-z ]',''),
                              n_bids = lambda x: pd.to_numeric(x.n_bids),
                              price = lambda x: pd.to_numeric(x.price))
                      .assign(sold_ind = lambda x: x.n_bids != 0)
                      .drop_duplicates()
                      .drop(['n_bids'],1)
                      .reset_index(drop=True))
    return munged


def vectorise_ad_text(raw_data, tf_min=0.01, tf_max=.9):
    word_counts = pd.Series((' '.join(raw_data.ad_title).split())).value_counts()
    lexicon = word_counts[word_counts.between(tf_min*len(raw_data),tf_max*len(raw_data))].index
    indicators = pd.DataFrame({w: raw_data.ad_title.str.contains(w) for w in lexicon})
    vectorised_data = pd.concat([raw_data, indicators],1).drop(['ad_title'],1)
    return vectorised_data

def get_training_data_arrays(data):
    X = data.drop('sold_ind',1).values.astype(np.float32)
    y = data['sold_ind'].values.astype(np.float32)
    return X,y


def train_model(X, y, **kwds):
    classifier = GradientBoostingClassifier(**kwds)
    classifier.fit(X,y)
    return classifier


def get_feature_importances(classifier, data):
    importances = pd.DataFrame({'importance': classifier.feature_importances_},
                                 index = data.drop('sold_ind',1).columns)
    return importances.sort_values(by='importance', ascending=False)


def score_ad(classifier, lexicon, price, ad_title):
    X = np.array([[price] + [w in ad_title for w in lexicon]])
    return classifier.predict_proba(X)[0][1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'This is a toy that was inspired by me getting bored with sifting through crap ads on ebay. It is a Python program which scrapes and prepares a dataset from ebay, trains a gradient boosted tree-ensemble model using scikit-learn, and then can predict the probability of the listing ending in a sale. The performance of the model is actually quite good, considering that it is only trained on the ad title and price. All the dependencies are included in the Anaconda distribution of Python 3.4. Example usage: `python ebay_sale_predictor.py --train --keywords "road+bicycle"` then `python ebay_sale_predictor.py --score --ad-title "vintage mens shimano" --price 900`')
    parser.add_argument("--train", help="train the model",
                    action="store_true")
    parser.add_argument("--score", help="score an ad",
                    action="store_true")
    parser.add_argument("--keywords", help="train mode only: search keywords used to build training dataset, use + for spaces")
    parser.add_argument("--ad-title", help="score mode only: ad text to be used for scoring")
    parser.add_argument("--price", help="train mode only: price of ad to be used for scoring")
    args = parser.parse_args()
    if args.train:
        if not args.keywords:
            print("Train mode missing required parameter --keywords")
        else:
            n_pages = 7
            search_term = args.keywords
            raw_data = pd.concat(parse_search_results(get_raw_search_results(search_term, page_number))
                     for page_number in range(1, n_pages + 1))
            munged_data = munge_data(raw_data)
            data = vectorise_ad_text(munged_data, tf_min=0.04, tf_max=0.8)
            X,y = get_training_data_arrays(data)
            classifier = train_model(X,y, n_estimators=150, min_samples_leaf=10, max_depth=5)
            lexicon = data.columns[2:]

            with open('classifier.pkl','wb') as f:
                pickle.dump((classifier, lexicon), f)

            roc_auc_cv = np.mean(cross_val_score(classifier,X,y, scoring = 'roc_auc', cv = StratifiedKFold(y, n_folds = 10)))
            print("roc_auc: {}".format(roc_auc_cv))

            print('Most important features:')
            print(get_feature_importances(classifier, data).iloc[0:10])


            #from cross_validated_roc import save_cv_roc
            #save_cv_roc(X, y, classifier)

    elif args.score:
        if not args.ad_title and args.price:
            print("Score mode missing required --ad_title and --price parameters")
        elif not os.path.isfile('classifier.pkl'):
            print("Classifier pickle not found, try running with --train mode first")
        else:
            with open('classifier.pkl','rb') as f:
                classifier,lexicon = pickle.load(f)
                print(score_ad(classifier, lexicon, int(args.price), args.ad_title))

    else:
        parser.print_help()
