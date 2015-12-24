This is a toy that was inspired by me getting bored with sifting through crap
ads on ebay. It is a Python program which scrapes and prepares a dataset from
ebay, trains a gradient boosted tree-ensemble model using scikit-learn, and
then can predict the probability of the listing ending in a sale. The
performance of the model is actually quite good, considering that it is only
trained on the ad title and price. All the dependencies are included in the
Anaconda distribution of Python 3.4. Example usage: `python
ebay_sale_predictor.py --train --keywords "road+bicycle"` then `python
ebay_sale_predictor.py --score --ad-title "vintage mens shimano" --price 900`

	optional arguments:
	  -h, --help           show this help message and exit
	  --train              train the model
	  --score              score an ad
	  --keywords KEYWORDS  train mode only: search keywords used to build training
			       dataset, use + for spaces
	  --ad-title AD_TITLE  score mode only: ad text to be used for scoring
	  --price PRICE        train mode only: price of ad to be used for scoring

