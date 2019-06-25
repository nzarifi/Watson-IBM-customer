#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:46:56 2019

@author: niloofarzarifi
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import os
os.getcwd()
os.chdir('/Users/niloofarzarifi/Desktop/Udacity/khaneh/Watson-IBM-customer/')
data = pd.read_csv('/Users/niloofarzarifi/Desktop/Udacity/khaneh/Watson-IBM-customer/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv')
data.head()
data.shape
data.isnull().sum()
sns.heatmap(data.isnull())  # full map of missing data
data.Policy.value_counts()
data.Policy.nunique()

col_list=data.columns.tolist()
DicOfCol={} 
for i in range(0,len(col_list)):
    case={col_list[i]:data[col_list[i]].nunique()} #most of the attributes have few labels. we can do lot of grouping!
    DicOfCol.update(case)
print ('unique values in each columns: ',DicOfCol ) 

#####____________________Time_______________########
#indicats that the dataset only shows months 1,2 in 2011. Therfore, we do not have season info
import datetime

def splitTime(df_name,column_name):
    df_name['year'] = pd.DatetimeIndex(df_name[column_name]).year
    df_name['month'] = pd.DatetimeIndex(df_name[column_name]).month
    df_name['day'] = pd.DatetimeIndex(df_name[column_name]).day

splitTime(data,'Effective To Date')
data.head()
data.year.value_counts()
data.month.value_counts()
data.day.value_counts()

#-----------------------------------------------------------------
#even the best engagement rate per day is less than 30%, 70% has Response=NO

Location_yes=data.loc[data['Response']=='Yes'] 
yes_renew=Location_yes.groupby(['day']).count()['Customer']
all_renew=data.groupby(['day']).count()['Customer']

rates_preRenewal=yes_renew/all_renew  

ax=(rates_preRenewal*100).plot(kind='bar',figsize=(7,7),color='blue',grid=True)
ax.set_ylabel('Engagement Rate per day%')
plt.savefig('day_yes',dpi=600,bbox_inches='tight')
plt.how()


##complementary plot with Response=No
Location_no=data.loc[data['Response']=='No'] 
no_renew=Location_no.groupby(['day']).count()['Customer']
all_renew=data.groupby(['day']).count()['Customer']

rates_preRenewal=no_renew/all_renew  

ax=(rates_preRenewal*100).plot(kind='bar',figsize=(7,7),color='blue',grid=True)
ax.set_ylabel('Engagement Rate per day%')
plt.savefig('day_no',dpi=600,bbox_inches='tight')
plt.how()

##----------------groupby practice--------------------
#these lines, similar answer
data.groupby(['EmploymentStatus']).count()['Customer']
data.groupby(by='EmploymentStatus', as_index=False).agg({'Customer': pd.Series.nunique})
data.groupby('EmploymentStatus')['Customer'].count()
data.groupby('EmploymentStatus')['Customer'].nunique()
data.Customer.groupby([data.EmploymentStatus.str.strip("'")]).nunique()
##check the difference
data.groupby(['EmploymentStatus', 'Customer']).count()
data.groupby(['Customer', 'EmploymentStatus']).count()

data.groupby('EmploymentStatus')['Education'].count() #Education is categorical ,not unique value
data.groupby('EmploymentStatus')['Education'].nunique()
#------------------crosstab---------------------------------
#Education and EmploymentStatus
crosstab=pd.crosstab(data.EmploymentStatus,data.Education,margins=False)
crosstab2=pd.crosstab(data.EmploymentStatus,data.Education,margins=True) #this one includes groupby result
#if I plot crosstab2 I have extra data related to 'All' or sum values of that group

crosstab.plot.bar(stacked=True)
plt.legend(title='Education')
plt.savefig('emp_Edu_stacked',dpi=600,bbox_inches='tight')
plt.show()

#Disabled, retired and medical leave are very short and we cannot see details so we stack and replot
stacked = crosstab.stack().reset_index().rename(columns={0:'value'})
# plot grouped bar chart
sns.barplot(x=stacked.EmploymentStatus, y=stacked.value, hue=stacked.Education)
plt.savefig('emp_Edu_unstacked',dpi=600,bbox_inches='tight')

#-----------employment and response-------------------------------------------
Emp_Response=pd.crosstab(data.EmploymentStatus,data.Response,margins=False)
Emp_Response.plot.bar(stacked=True)

plt.savefig('emp_Res',dpi=600,bbox_inches='tight')
plt.legend(title='Employment_Response')



#education and response
Edu_Response=pd.crosstab(data.Education,data.Response,margins=True) #margins=True let us to have extra col=All
Edu_Response.plot.bar(stacked=True)
plt.savefig('Edu_Res',dpi=600,bbox_inches='tight')
plt.legend(title='Education') #absolute values we need percentage to compare groups

Edu_Response.columns.tolist()

#so far we only checked the frequency how about the ratio?
df_total = Edu_Response['All']
df_rel = Edu_Response[Edu_Response.columns[0:]].div(df_total, 0)*100 #generate percentage
df_rel
df_rel=df_rel.drop(['All'], axis=1)
df_rel.plot.bar(stacked=True) #not significant difference between groups
plt.savefig('Edu_Res_percentage',dpi=600,bbox_inches='tight')

#here we shows values on the bar
ax=df_rel.plot(kind='bar', stacked=True, colormap="summer")
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    ax.annotate('{:.0%}'.format(height/100), (p.get_x() + .15 * width, p.get_y() + .4 * height))
    plt.xticks(fontsize=10, rotation=60)
    ax.set_title('Engagement percentage', fontsize=14 )
    
plt.savefig('Edu_Res_percentage_label', bbox_inches='tight',dpi=600)    
plt.show()
#replace the following line if you don not want to show "%"
#ax.annotate('{:.0f}'.format(height), (p.get_x() + .15 * width, p.get_y() + .4 * height))

df_rel.round(0)  #rounds the data frame without decimal 

#------------------------------------------------------------------------------
#Sales Channel succeess vs education 
Location_yes

Sale_Edu=pd.crosstab(Location_yes.Education,Location_yes['Sales Channel'],margins=True) 
Total=pd.crosstab(data.Education,data['Sales Channel'],margins=True) 

rates=(Sale_Edu/Total)*100

ax=(rates).plot(kind='bar',figsize=(7,7),colormap='rainbow',grid=True)
ax.set_ylabel('Engagement Rate%')
ax.set_title("Sales success to get Response='Yes' over all Yes/No responses")
plt.savefig('Sale_Edu', bbox_inches='tight',dpi=600)   
plt.show() #apparently 'agent' is more successful except PhD group' and call center is less efficient 


#Sales Channel succeess vs employment 
Location_yes

Sale_Emp=pd.crosstab(Location_yes.EmploymentStatus,Location_yes['Sales Channel'],margins=False) 
Total=pd.crosstab(data.EmploymentStatus,data['Sales Channel'],margins=False) 

rates=(Sale_Emp/Total)*100

ax=(rates).plot(kind='bar',figsize=(7,7),colormap='rainbow',grid=True)
ax.set_ylabel('Engagement Rate%')
ax.set_title('Sales success to get Response="Yes" over all Yes/No responses')
plt.savefig('Sale_Emp', bbox_inches='tight',dpi=600) 

#now plot all individual Sale channel 
ax=rates.plot(kind='bar',figsize=(7,7),subplots=True, layout=(2,2),colormap='rainbow',grid=True)
plt.savefig('Sale_Emp_rate_indivi', bbox_inches='tight',dpi=600)
#how about frequency
ax=(Sale_Emp).plot(kind='bar',figsize=(7,7),colormap='rainbow',grid=True)
ax.set_ylabel('Engagement number')
ax.set_title('Sales success to get Response="Yes"')
plt.savefig('Sale_Emp_freq', bbox_inches='tight',dpi=600) 
#for employed : Engagement Rate% Agent efficiency is almost double 

#now plot all individual Sale channel 
ax=Sale_Emp.plot(kind='bar',figsize=(7,7),subplots=True, layout=(2,2),colormap='rainbow',grid=True)
plt.savefig('Sale_Emp_freq_indivi', bbox_inches='tight',dpi=600)



##two Sale_Emp percentage and frequency beside each other :)

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
fig, axes = plt.subplots(ncols=2)
Sale_Emp.plot(kind='bar',figsize=(9,6),colormap='rainbow',grid=True,ax=axes[1])
rates.plot(kind='bar',figsize=(9,6),colormap='rainbow',grid=True,log=False, ax=axes[0])

##---------------------------------------------------------------
