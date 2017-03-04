import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Reading training and test data
train = pd.read_json('../../Rental_Listing_Inquiries_Data/train.json')
test = pd.read_json('../../Rental_Listing_Inquiries_Data/test.json')
print train.shape, test.shape


# Plotting interest level with price - one of the most intuitive predictor
train.plot('interest_level', 'price', kind='line')
plt.show()


# Checking if any data is missing
print train.isnull().values.any()
print test.isnull().values.any()

