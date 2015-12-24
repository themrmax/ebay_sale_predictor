from ebay_sale_predictor import *
from nose.tools import *
from pandas.util.testing import assert_frame_equal
import pandas as pd
import numpy as np


def test_parse_search_results():
    raw_text = 'src="http://thumbs.ebaystatic.com/images/g/1OIAAOSwA4dWL-XA/s-l225.jpg" class="img" alt=\'BMC SSX 01 Road Bike\' />\n\t\t\t\t</a>\n\t\t\t</div></div>\n\t<h3 class="lvtitle"><a href="http://www.ebay.com.au/itm/BMC-SSX-01-Road-Bike-/301785728725?hash=item4643d4c6d5:g:1OIAAOSwA4dWL-XA"  class="vip" title="Click this link to access BMC SSX 01 Road Bike">BMC SSX 01 Road Bike</a>\n\t\t</h3>\n\t<div class="lvsubtitle" >\n\t\tBMC</div>\n<ul class="lvprices left space-zero">\n\n\t<li class="lvprice prc">\n\t\t\t<span  class="bold bidsold"><b>AU $</b>790.00</span>\n</li>\n\t\t<li class="lvformat">\n\t\t\t<span >30 bids</span>\n'
    expected = pd.DataFrame({'ad_title':['BMC SSX 01 Road Bike'],'n_bids': ['30'], 'price':['790']})
    actual = parse_search_results(raw_text)
    assert_frame_equal(actual.sort_index(1), expected.sort_index(1))

@raises(Exception)
def test_parse_search_results_raises_exception_when_bids_missing():
    raw_text = 'src="http://thumbs.ebaystatic.com/images/g/1OIAAOSwA4dWL-XA/s-l225.jpg" class="img" alt=\'BMC SSX 01 Road Bike\' />\n\t\t\t\t</a>\n\t\t\t</div></div>\n\t<h3 class="lvtitle"><a href="http://www.ebay.com.au/itm/BMC-SSX-01-Road-Bike-/301785728725?hash=item4643d4c6d5:g:1OIAAOSwA4dWL-XA"  class="vip" title="Click this link to access BMC SSX 01 Road Bike">BMC SSX 01 Road Bike</a>\n\t\t</h3>\n\t<div class="lvsubtitle" >\n\t\tBMC</div>\n<ul class="lvprices left space-zero">\n\n\t<li class="lvprice prc">\n\t\t\t<span  class="bold bidsold"><b>AU $</b>790.00</span>\n</li>\n\t\t<li class="lvformat">\n\t\t\t'
    df = parse_search_results(raw_text)


def test_munge_data():
    raw_data = pd.DataFrame({'ad_title':['BMC SSX 01 Road Bike'],'n_bids': ['30'], 'price':['790']})
    expected = pd.DataFrame({'ad_title':['bmc ssx  road bike'],'price':[790],'sold_ind':True})
    actual = munge_data(raw_data)
    assert_frame_equal(actual.sort_index(1), expected.sort_index(1))


def test_vectorise_ad_text():
    data = pd.DataFrame({'ad_title':['bmc ssx  road bike'],'price':[790],'sold_ind':True})
    expected = pd.DataFrame({'price':[790], 'sold_ind': [True], 'bike': [True], 'bmc': [True], 'road': [True], 'ssx': [True]})
    actual = vectorise_ad_text(data, tf_min = 0, tf_max=1)
    assert_frame_equal(actual.sort_index(1), expected.sort_index(1))

def test_get_training_data_arrays():
    data = pd.DataFrame({'bike': [True], 'bmc': [True],'price':[790], 'road': [True], 'ssx': [True],'sold_ind': [True]})
    expected_X = np.array([1.0,1.0,790.0,1.0,1.0], dtype = np.float32)
    expected_y = np.array([1.0], dtype = np.float32)
    actual_X, actual_y = get_training_data_arrays(data)
    assert (actual_X == expected_X).all() and (actual_y == expected_y).all()
