#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress


# In[35]:


df = pd.read_csv(r'c:/users/srinivas/Overload Ware Labs AI/Data/Train.csv')


# In[36]:


df


# To see the first 5 rows

# In[37]:


df.head()


# To Check the number of rows and columns - (How big and To find out Column types and missing values)

# In[38]:


df.shape # looking at the shape of data


# To show summary of columns and datatypes

# In[39]:


df.info()  # taking a look at info of the data.


# To Check for missing values in each column

# In[40]:


df.isnull().sum() # checking for null values using missingno module


# In[81]:


df.describe() # getting description of data


# In[82]:


# heatmap of the data for checking the correlation between the features and target column.
plt.figure(figsize = (18, 7))
sns.heatmap(df.corr(numeric_only=True), annot = True, fmt = '0.2f', annot_kws = {'size' : 15}, linewidth = 5, linecolor = 'Red')
plt.show()


# #Quick numeric overview

# In[ ]:





# In[83]:


Exploratory Data Analysis (EDA)


# Lets Analyze the given data

# Q1: What % of shipments are delayed?

# In[84]:


# 0 = Delayed, 1 = On Time (precise insight - float64)
delayed_count = df['Reached.on.Time_Y.N'].value_counts(normalize=True) * 100 


# In[85]:


delayed_count


# In[86]:


# On-time vs Delayed counts
delivery_counts = df['Reached.on.Time_Y.N'].value_counts()
labels = ['On Time', 'Delayed']
colors = ['green', 'red']

plt.figure(figsize=(6, 6))
plt.pie(delivery_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Delivery Status Distribution')
plt.axis('equal')
plt.show()


# Q2: Which shipping mode has the highest delay rate? (Road vs ship vs flight)

# In[87]:


#To Filter only delayed orders
delayed_orders = df[df['Reached.on.Time_Y.N'] == 0]

#To Count delays by shipping mode
shipment_mode_delay = delayed_orders['Mode_of_Shipment'].value_counts(normalize=True) * 100
shipment_mode_delay
shipment_mode_delay.round(2)


# In[88]:


delayed_df = df[df['Reached.on.Time_Y.N'] == 0]
shipment_mode_delay = delayed_df['Mode_of_Shipment'].value_counts(normalize=True) * 100

shipment_mode_delay.plot(kind='bar',figsize=(8, 5))
plt.ylabel('Percentage of Delayed Deliveries')
plt.title('Delay Distribution by Mode of Shipment')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()


# Q3: high-priority products get delayed more or less?

# In[89]:


# Delay rate by product importance
delay_by_importance = delayed_orders['Product_importance'].value_counts(normalize=True) * 100
delay_by_importance.round(2)


# In[90]:


importance_counts = delayed_df['Product_importance'].value_counts(normalize=True) * 100

importance_counts.plot(kind='pie', autopct='%1.1f%%', figsize=(6, 6))
plt.title('Product Importance Among Delayed Deliveries')
plt.ylabel('')
plt.show()


# Q4: Does higher discounts lead to more delivery delays?

# In[91]:


# Average discount for on-time vs delayed deliveries
df.groupby('Reached.on.Time_Y.N')['Discount_offered'].mean().round(2)


# In[92]:


df.groupby('Reached.on.Time_Y.N')['Discount_offered'].mean().plot(kind='bar', figsize=(6, 4))
plt.xticks(ticks=[0,1], labels=['Delayed', 'On Time'], rotation=0)
plt.ylabel('Average Discount Offered')
plt.title('Discount vs Delivery Status')
plt.grid(axis='y')
plt.show()


# Q5: To find out which warehouses have the most delivery delays?

# In[93]:


warehouse_delay_counts = delayed_orders['Warehouse_block'].value_counts()
warehouse_delay_counts


# In[94]:


delayed_df['Warehouse_block'].value_counts().plot(kind='bar', figsize=(6, 4))
plt.title('Delays by Warehouse Block')
plt.xlabel('Warehouse')
plt.ylabel('Number of Delays')
plt.grid(axis='y')
plt.show()


# Q6: What was the Customer Rating? And was the product delivered on time?

# In[95]:


sns.barplot(x='Customer_rating', y='Reached.on.Time_Y.N', data=df)
plt.title('Customer Rating vs On-Time Delivery Rate')
plt.xlabel('Customer Rating (1=worst, 5=best)')
plt.ylabel('Proportion Delivered On Time')
plt.show()


# Does customer calls affect ratings?

# In[96]:


# making a lineplot to check the relation between customer care calls, customer ratings and gender

plt.figure(figsize = (18, 9))
sns.lineplot(x='Customer_care_calls', y='Customer_rating', hue='Gender', data=df, errorbar=('ci', 0))
plt.title('Relation between Customer Care Calls and Customer Rating of Males and Females\n',
          fontsize = 15)
plt.show()


# Q7: Is Customer Query Being Answered?

# In[97]:


#More calls → potential delivery issues or product dissatisfaction.
sns.boxplot(x='Reached.on.Time_Y.N', y='Customer_care_calls', data=df)
plt.title('Customer Care Calls vs Delivery Status')
plt.xlabel('Delivery On Time (1=Yes, 0=No)')
plt.ylabel('Number of Customer Care Calls')
plt.show()


# Q8: If Product Importance is High, does it have Highest Ratings or Get Delivered On Time?

# a. Product Importance vs On-Time Delivery:

# In[98]:


sns.barplot(x='Product_importance', y='Reached.on.Time_Y.N', data=df)
plt.title('Product Importance vs On-Time Delivery Rate')
plt.xlabel('Product Importance')
plt.ylabel('Proportion Delivered On Time')
plt.show()


# b. Product Importance vs Customer Rating:

# In[99]:


sns.boxplot(x='Product_importance', y='Customer_rating', data=df)
plt.title('Customer Rating by Product Importance')
plt.xlabel('Product Importance')
plt.ylabel('Customer Rating')
plt.show()


# In[100]:


# making a distplot of cost of the product column

plt.figure(figsize = (15, 7))
ax = sns.histplot(df['Cost_of_the_Product'], bins = 100, color = 'y')

plt.show()


# In[101]:


# making a distplot of discount offered column

plt.figure(figsize = (15, 7))
ax = sns.histplot(df['Discount_offered'], color = 'y')

plt.show()


# In[102]:


plt.figure(figsize = (15, 7))
ax = sns.histplot(df['Weight_in_gms'], bins = 100, color = 'y')

plt.show()


# Which type of warehouse contains most weights ?

# In[103]:


# creating a dataframe of warehouse block and weights in gram columns 

ware_block_weight = df.groupby(['Warehouse_block'])['Weight_in_gms'].sum().reset_index()
ware_block_weight


# Which mode of shipmemnt carries most weights ?

# In[104]:


shipment_mode_weight = df.groupby(['Mode_of_Shipment'])['Weight_in_gms'].sum().reset_index()
shipment_mode_weight


# Effect of Warehouse on Cost of Product ?

# In[105]:


warehouse_weight = df.groupby(['Warehouse_block'])['Cost_of_the_Product'].sum().reset_index()
warehouse_weight


# In[106]:


#creating scatter plot to see the relation between cost of the product and the discount offered and the relation with
# whether or not th product will reach on time
plt.figure(figsize = (15, 7))
sns.scatterplot(x='Discount_offered', y='Cost_of_the_Product', data=df, hue='Reached.on.Time_Y.N')

plt.show()


# In[ ]:





# In[ ]:




