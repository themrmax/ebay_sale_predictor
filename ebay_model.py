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



def get_parsed_search_results(search_term, page_number):
    t = get_raw_search_results(search_term, page_number)
    adtext = re.findall("alt='([^']*)", t) #don't parse html with regex because tony the pony
    n_bids = re.findall("(\d+) bid", t)
    price = re.findall("\$</b>(\d+)", t)

    if not (len(adtext) == len(n_bids) and len(n_bids) == len(price)):
        raise Exception("Parsing failed, length mismatch between ad text, bids and price vectors")
    return pd.DataFrame({'adtext':adtext, 'n_bids':n_bids, 'price':price})


def get_raw_search_results(search_term, page_number):
    url = 'http://www.ebay.com.au/sch/i.html?_fcid=15&LH_Auction=1&LH_Complete=1&_clu=2&gbr=1&_from=R40&_sacat=0&_nkw={}&_ipg=200&rt=nc&_pgn='.format(search_term)
    return requests.get(url+str(page_number)).text


def get_ebay_data(search_term='retro+bicycle', n_pages=2):
    raw_data = pd.concat(get_parsed_search_results(search_term, page_number)
                     for page_number in range(1, n_pages + 1))
    munged = (raw_data.assign(adtext = lambda x: x.adtext.str.lower().str.replace('[^a-z ]',''))
                     .convert_objects(convert_numeric=True)
                     .assign(sold_ind = lambda x: x.n_bids != 0)
                     .drop_duplicates()
                     .reset_index(drop=True))
    return munged


def vectorise_data(raw_data, tf_min=0.01, tf_max=.9):
    word_counts = pd.Series((' '.join(raw_data.adtext).split())).value_counts()
    lexicon = word_counts[word_counts.between(tf_min*len(raw_data),tf_max*len(raw_data))].index
    indicators = pd.DataFrame({w: raw_data.adtext.str.contains(w) for w in lexicon})
    vectorised_data = pd.concat([raw_data, indicators],1).drop(['adtext','n_bids'],1)
    return vectorised_data


def train_model(vectorised_data, **kwds):
    X = data.drop('sold_ind',1).values
    y = data['sold_ind']
    classifier = GradientBoostingClassifier(**kwds)
    classifier.fit(X,y)
    return classifier


def get_feature_importances(classifier, data):
    importances = pd.DataFrame({'importance': classifier.feature_importances_},
                                 index = data.drop('sold_ind',1).columns)
    return importances.sort_values(by='importance', ascending=False)


def score_ad(classifier, lexicon, price, adtext):
    X = np.array([[price] + [w in adtext for w in lexicon]])
    return classifier.predict_proba(X)[0][1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Train a model or score against a trained model to predict the probablility of an ebay listing ending in a sale. Example usage: `python ebay_model.py --train --keywords "road+bicycle"` or `python ebay_model.py --score --adtext "vintage mens shimano" --price 900`')
    parser.add_argument("--train", help="train the model",
                    action="store_true")
    parser.add_argument("--score", help="score an ad",
                    action="store_true")
    parser.add_argument("--keywords", help="train mode only: search keywords used to build training dataset, use + for spaces")
    parser.add_argument("--adtext", help="score mode only: ad text to be used for scoring")
    parser.add_argument("--price", help="train mode only: price of ad to be used for scoring")
    args = parser.parse_args()
    if args.train:
        if not args.keywords:
            print("Train mode missing required parameter --keywords")
        else:
            web_data = get_ebay_data(search_term=args.keywords, n_pages=7)
            data = vectorise_data(web_data, tf_min=0.04, tf_max=0.8)
            classifier = train_model(data, n_estimators=150, min_samples_leaf=10, max_depth=5)
            lexicon = data.columns[2:]
            with open('classifier.pkl','wb') as f:
                pickle.dump((classifier, lexicon), f)

            X = data.drop('sold_ind',1).values
            y = data['sold_ind']
            roc_auc_cv = np.mean(cross_val_score(classifier,X,y, scoring = 'roc_auc', cv = StratifiedKFold(y, n_folds = 10)))
            print("roc_auc: {}".format(roc_auc_cv))

            print('Most important features:')
            print(get_feature_importances(classifier, data).iloc[0:10])


            from cross_validated_roc import save_cv_roc
            save_cv_roc(X, y, classifier)

    elif args.score:
        if not args.adtext and args.price:
            print("Score mode missing required --adtext and --price parameters")
        elif not os.path.isfile('classifier.pkl'):
            print("Classifier pickle not found, try running with --train mode first")
        else:
            with open('classifier.pkl','rb') as f:
                classifier,lexicon = pickle.load(f)
                print(score_ad(classifier, lexicon, int(args.price), args.adtext))

    else:
        parser.print_help()
