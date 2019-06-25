#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 20:40:33 2019
I do not know author's name I replotted and added extra plots here
@author: niloofarzarifi
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import os
os.getcwd()

data = pd.read_csv('/Users/niloofarzarifi/Desktop/Udacity/khaneh/Watson-IBM-customer/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv')
data.head()
data.shape
data.columns
data.isnull().sum()
data.Policy.value_counts()
data.Policy.nunique()

#customer response
data.groupby('Response').count()['Customer']
ax=data.groupby('Response').count()['Customer'].plot(kind='bar',color='red',grid=True,
               figsize=(7,7),title='Marketing Engagement')
ax.set_xlabel('Engaged')
ax.set_ylabel('Count')
plt.savefig('wc-Yes-No',dpi=600,bbox_inches='tight')
plt.show()
# % of engagement
data.groupby('Response').count()['Customer']/data.shape[0]
#---------------------------------------------------
#engagement rates per renewal offer type

data[['Customer','Renew Offer Type','Response']]
data.Customer.nunique() #all Customers are Unique
Location_yes=data.loc[data['Response']=='Yes']  #shows Yes response
Yes_renew=Location_yes.groupby(['Renew Offer Type']).count()['Customer'] #counts renew offer plus 'Yes'
all_renew=data.groupby(['Renew Offer Type']).count()['Customer'] #counts all Yes/no renew offer type

#shows pernentage of renew offer with Yes response
rates_preRenewal=Yes_renew/all_renew  #offer4 has zero 'Yes'!

ax=(rates_preRenewal*100).plot(kind='bar',figsize=(7,7),color='blue',grid=True)
ax.set_ylabel('Engagement Rate%')
plt.savefig('wc-offer-type',dpi=600,bbox_inches='tight')
plt.how()

#######try to plot pie chart Renew Offer Type
fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

recipe = ["3752  Offer1",
          "2926  Offer2",
          "1432  Offer3",
          "1024  Offer4"]

data = [float(x.split()[0]) for x in recipe]
ingredients = [x.split()[-1] for x in recipe]

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d})".format(pct, absolute)


wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"))
ax.legend(wedges, ingredients,
          title="all offers",
          loc="best",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=8, weight="bold")
ax.set_title("Renew Offer Type")
plt.savefig('wc-pie_allOffer.png', bbox_inches='tight',dpi=600)
plt.show()

#-----------------------------------------------------
#offer Type vs vehicle type and engagement rate
by_offer_type_data=Location_yes.groupby(['Renew Offer Type', 'Vehicle Class']).count()['Customer']/all_renew

by_offer_type_data = by_offer_type_data.unstack().fillna(0)



ax = (by_offer_type_data*100.0).plot(
    kind='bar',
    figsize=(10, 7),
grid=True )
ax.set_ylabel('Engagement Rate (%)')
plt.savefig('wc-offer-vehicle-rate.png', bbox_inches='tight',dpi=600)
plt.show()
#--------------------------------------------------------------------------
#Engagement Rates by Sales Channel

by_sales_channel_df = data.loc[data['Response'] == 'Yes'
                               ].groupby(['Sales Channel']).count()['Customer'
                               ]/data.groupby(['Sales Channel']).count()['Customer']

ax = (by_sales_channel_df*100.0).plot(kind='bar',figsize=(7, 7), color='palegreen', grid=True)
ax.set_ylabel('Engagement Rate (%)')
plt.savefig('wc-eng-sales.png', bbox_inches='tight',dpi=600)
plt.show()



#Sales Channel & Vehicle Size
by_sales_channel_df = data.loc[data['Response'] == 'Yes'].groupby(['Sales Channel', 'Vehicle Size'
                              ]).count()['Customer'] / data.groupby('Sales Channel').count()['Customer']

by_sales_channel_df = by_sales_channel_df.unstack().fillna(0)

ax = (by_sales_channel_df*100.0).plot(kind='bar',figsize=(10, 7),grid=True )
ax.set_ylabel('Engagement Rate (%)')
plt.savefig('wc-eng-vehicle.png', bbox_inches='tight',dpi=600)
plt.show()
#--------------------------------------------------------------------
#Engagement Rates by Months Since Policy Inception
by_months_since_inception_df = data.loc[data['Response'] == 'Yes'
                                      ].groupby(by='Months Since Policy Inception')['Response'
                                      ].count() / data.groupby(by='Months Since Policy Inception')['Response'
                                      ].count() * 100.0

by_months_since_inception_df.fillna(0)

ax = by_months_since_inception_df.fillna(0).plot(figsize=(10, 7),grid=True,color='skyblue')
ax.set_title=('Engagement Rates by Months Since Inception')
ax.set_xlabel('Months Since Policy Inception')
ax.set_ylabel('Engagement Rate (%)')
plt.savefig('wc-eng-month.png', bbox_inches='tight',dpi=600)
plt.show()

#-------------------------------------------------------------------------------------------
##Customer Segmentation by CLV & Months Since Policy Inception
data['Customer Lifetime Value'].describe()

data['CLV Segment'] = data['Customer Lifetime Value'].apply(
lambda x: 'High' if x > data['Customer Lifetime Value'].median() else 'Low')

data['Policy Age Segment'] = data['Months Since Policy Inception'].apply(
lambda x: 'High' if x > data['Months Since Policy Inception'].median() else 'Low')


data.head()

# Visualize these segments
ax = data.loc[
    (data['CLV Segment'] == 'High') & (data['Policy Age Segment'] == 'High')
    ].plot.scatter(x='Months Since Policy Inception', y='Customer Lifetime Value', logy=True,color='red')


data.loc[(data['CLV Segment'] == 'Low') & (data['Policy Age Segment'] == 'High'
         )].plot.scatter(ax=ax,x='Months Since Policy Inception', 
         y='Customer Lifetime Value',logy=True,color='blue')

data.loc[(data['CLV Segment'] == 'High') & (data['Policy Age Segment'] == 'Low'
         )].plot.scatter(ax=ax,x='Months Since Policy Inception', y='Customer Lifetime Value', logy=True,
          color='orange')
data.loc[(data['CLV Segment'] == 'Low') & (data['Policy Age Segment'] == 'Low')].plot.scatter(
         ax=ax,x='Months Since Policy Inception',y='Customer Lifetime Value', logy=True,
         color='green',grid=True,figsize=(10, 7))
         
ax.set_ylabel('CLV (in log scale)')
ax.set_xlabel('Months Since Policy Inception')
ax.set_title('Segments by CLV and Policy Age')
plt.savefig('wc-seg.png', bbox_inches='tight',dpi=600)
plt.show()
#-----------------------------------------------------------------------------------
# See whether there is any noticeable difference in the engagement rates

engagement_rates_by_segment_df = data.loc[data['Response'] == 'Yes'].groupby([
        'CLV Segment', 'Policy Age Segment']). count()['Customer'] / data.groupby([
        'CLV Segment', 'Policy Age Segment']).count()['Customer']


engagement_rates_by_segment_df

ax = (engagement_rates_by_segment_df.unstack()*100.0).plot(kind='bar',
             figsize=(10, 7),grid=True )
ax.set_ylabel('Engagement Rate (%)')
ax.set_title('Engagement Rates by Customer Segments')
plt.savefig('wc-L-H.png', bbox_inches='tight',dpi=600)
plt.show()

##############################
###########################
#####################
