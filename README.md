#ebay sale predictor

A little toy project which scrapes and prepares a dataset from
ebay, trains a scikit-learn Gradient Boosting Classifier on the price and ad title, and
tries to predict the probability of the listing ending in a sale. The
performance of the model is actually quite good, considering that it is only
trained on the ad title and price. 

## Dependencies 

All the dependencies should be included in theAnaconda distribution of Python 3.4. 

## Test

    nosetests

## Example usage 

    python ebay_sale_predictor.py --train --keywords "road+bicycle"
    python ebay_sale_predictor.py --score --ad-title "vintage mens shimano" --price 900



