# Capstone DataScience Project

Repo for my Capstone Project in DataScience Bootcamp 2026

Material analysis of cultural heritage artifacts can help understand their history. Often the identity or origin is unknown or lost. 
During colonization in India, palm-leaf manuscripts were relocated and gathered without any information about their origin. 
Today, libraries take care of this treasure and in many cases there is little information preserved.

It was shown that the manuscripts show different patterns in infrared spectra due to changes in their molecular composition based on e.g., their age, preservation status, species and origin. 
Nevertheless, the classification is not straightforward, requires preprocessing and was not successful for the data set.
A more sophisticated approach using CNN is proposed, turning the spectral data into an 1D-image which preserves the spatial information. Also other supervised ML models can be tested.
The final result will be a model that can classify the spectra to a manuscripts identity, origin or type.

## Data
The data can be found here: https://doi.org/10.25592/uhhfdm.16468

## Software
Python 3.11.3
please refer to requirements.txt for packages used.