from ebay_sale_predictor import *
from nose.tools import *
from pandas.util.testing import assert_frame_equal
import pandas as pd

def test_parse_search_results():
    raw_text = 'src="http://thumbs.ebaystatic.com/images/g/1OIAAOSwA4dWL-XA/s-l225.jpg" class="img" alt=\'BMC SSX 01 Road Bike\' />\n\t\t\t\t</a>\n\t\t\t</div></div>\n\t<h3 class="lvtitle"><a href="http://www.ebay.com.au/itm/BMC-SSX-01-Road-Bike-/301785728725?hash=item4643d4c6d5:g:1OIAAOSwA4dWL-XA"  class="vip" title="Click this link to access BMC SSX 01 Road Bike">BMC SSX 01 Road Bike</a>\n\t\t</h3>\n\t<div class="lvsubtitle" >\n\t\tBMC</div>\n<ul class="lvprices left space-zero">\n\n\t<li class="lvprice prc">\n\t\t\t<span  class="bold bidsold"><b>AU $</b>790.00</span>\n</li>\n\t\t<li class="lvformat">\n\t\t\t<span >30 bids</span>\n'
    assert_frame_equal(parse_search_results(raw_text), pd.DataFrame({'ad_title':['BMC SSX 01 Road Bike'],'n_bids': ['30'], 'price':['790']}))

@raises(Exception)
def test_parse_search_results_raises_exception_when_bids_missing():
    raw_text = 'src="http://thumbs.ebaystatic.com/images/g/1OIAAOSwA4dWL-XA/s-l225.jpg" class="img" alt=\'BMC SSX 01 Road Bike\' />\n\t\t\t\t</a>\n\t\t\t</div></div>\n\t<h3 class="lvtitle"><a href="http://www.ebay.com.au/itm/BMC-SSX-01-Road-Bike-/301785728725?hash=item4643d4c6d5:g:1OIAAOSwA4dWL-XA"  class="vip" title="Click this link to access BMC SSX 01 Road Bike">BMC SSX 01 Road Bike</a>\n\t\t</h3>\n\t<div class="lvsubtitle" >\n\t\tBMC</div>\n<ul class="lvprices left space-zero">\n\n\t<li class="lvprice prc">\n\t\t\t<span  class="bold bidsold"><b>AU $</b>790.00</span>\n</li>\n\t\t<li class="lvformat">\n\t\t\t'
    df = parse_search_results(raw_text)


def test_munge_data():
    raw_data = pd.DataFrame({'ad_title':['BMC SSX 01 Road Bike'],'n_bids': ['30'], 'price':['790']})

    assert_frame_equal(munge_data(raw_data), pd.DataFrame({'ad_title':['bmc ssx  road bike'],'n_bids':[30],'price':[790],'sold_ind':True}))




