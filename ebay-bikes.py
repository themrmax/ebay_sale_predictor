import requests
import re
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt

def get_parsed_search_results(search_term, page_number):
    t = get_raw_search_results(search_term, page_number)
    ad_text = re.findall("alt='([^']*)", t) #don't parse html with regex because tony the pony
    n_bids = re.findall("(\d+) bid", t)
    price = re.findall("\$</b>(\d+)", t)
    return pd.DataFrame({'ad_text':ad_text, 'n_bids':n_bids, 'price':price})

def get_raw_search_results(search_term, page_number):
    url = 'http://www.ebay.com.au/sch/i.html?_fcid=15&LH_Auction=1&LH_Complete=1&_clu=2&gbr=1&_from=R40&_sacat=0&_nkw={}&_ipg=200&rt=nc&_pgn='.format(search_term)
    return requests.get(url+str(page_number)).text

def get_ebay_data(search_term='retro+bicycle', n_pages=2):
    raw_data = pd.concat(get_parsed_search_results(search_term, page_number)
                     for page_number in range(1, n_pages + 1))
    munged = raw_data.assign(ad_text = lambda x: x.ad_text.str.lower().str.replace('[^a-z ]',''))\
                     .convert_objects(convert_numeric=True)\
                     .assign(sold_ind = lambda x: x.n_bids != 0)\
                     .drop_duplicates()\
                     .reset_index(drop=True)
    return munged

def vectorise_data(raw_data, tf_min=0.01, tf_max=.9):
    word_counts = pd.Series((' '.join(raw_data.ad_text).split())).value_counts()
    lexicon = word_counts[word_counts.between(tf_min*len(raw_data),tf_max*len(raw_data))].index
    indicators = pd.DataFrame({w: raw_data.ad_text.str.contains(w) for w in lexicon})

    vectorised_data = pd.concat([raw_data, indicators],1).drop(['ad_text','n_bids'],1)
    return vectorised_data

def train_model(vectorised_data):
    X = data.drop('sold_ind',1).values
    y = data['sold_ind']

    classifier = GradientBoostingClassifier(n_estimators = 150,
                                            min_samples_leaf = 1,
                                            max_depth = 10)
    classifier.fit(X,y)
    return classifier

def get_feature_importances(classifier, data):
    importances = pd.DataFrame({'importance': classifier.feature_importances_},
                                 index = data.drop('sold_ind',1).columns)


    return importances.sort_values(by='importance', ascending=False)



web_data = get_ebay_data(search_term='road+bicycle', n_pages=7)
data = vectorise_data(web_data, tf_min=0.05, tf_max=0.8)
classifier = train_model(data)

print('Most important features:')
print(get_feature_importances(classifier, data).iloc[0:10])

from cross_validated_roc import save_cv_roc
save_cv_roc(data, model)

