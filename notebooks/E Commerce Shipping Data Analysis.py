#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress


# In[71]:


df = pd.read_csv(r'c:/users/srinivas/Overload Ware Labs AI/Data/Train.csv')


# In[72]:


df


# To see the first 5 rows

# In[73]:


df.head()


# To Check the number of rows and columns - (How big and To find out Column types and missing values)

# In[74]:


df.shape


# To show summary of columns and datatypes

# In[75]:


df.info()


# To Check for missing values in each column

# In[76]:


df.isnull().sum()


# #Quick numeric overview

# In[77]:


df.describe()


# Lets Analyze the given data

# Q1: What % of shipments are delayed?

# In[78]:


# 0 = Delayed, 1 = On Time (precise insight - float64)
delayed_count = df['Reached.on.Time_Y.N'].value_counts(normalize=True) * 100 


# In[79]:


delayed_count


# In[80]:


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

# In[81]:


#To Filter only delayed orders
delayed_orders = df[df['Reached.on.Time_Y.N'] == 0]

#To Count delays by shipping mode
shipment_mode_delay = delayed_orders['Mode_of_Shipment'].value_counts(normalize=True) * 100
shipment_mode_delay
shipment_mode_delay.round(2)


# In[82]:


delayed_df = df[df['Reached.on.Time_Y.N'] == 0]
shipment_mode_delay = delayed_df['Mode_of_Shipment'].value_counts(normalize=True) * 100

shipment_mode_delay.plot(kind='bar',figsize=(8, 5))
plt.ylabel('Percentage of Delayed Deliveries')
plt.title('Delay Distribution by Mode of Shipment')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()


# Q3: high-priority products get delayed more or less?

# In[83]:


# Delay rate by product importance
delay_by_importance = delayed_orders['Product_importance'].value_counts(normalize=True) * 100
delay_by_importance.round(2)


# In[84]:


importance_counts = delayed_df['Product_importance'].value_counts(normalize=True) * 100

importance_counts.plot(kind='pie', autopct='%1.1f%%', figsize=(6, 6))
plt.title('Product Importance Among Delayed Deliveries')
plt.ylabel('')
plt.show()


# Q4: Does higher discounts lead to more delivery delays?

# In[85]:


# Average discount for on-time vs delayed deliveries
df.groupby('Reached.on.Time_Y.N')['Discount_offered'].mean().round(2)


# In[86]:


df.groupby('Reached.on.Time_Y.N')['Discount_offered'].mean().plot(kind='bar', figsize=(6, 4))
plt.xticks(ticks=[0,1], labels=['Delayed', 'On Time'], rotation=0)
plt.ylabel('Average Discount Offered')
plt.title('Discount vs Delivery Status')
plt.grid(axis='y')
plt.show()


# Q5: To find out which warehouses have the most delivery delays?

# In[87]:


warehouse_delay_counts = delayed_orders['Warehouse_block'].value_counts()
warehouse_delay_counts


# In[88]:


delayed_df['Warehouse_block'].value_counts().plot(kind='bar', figsize=(6, 4))
plt.title('Delays by Warehouse Block')
plt.xlabel('Warehouse')
plt.ylabel('Number of Delays')
plt.grid(axis='y')
plt.show()


# Q6: What was the Customer Rating? And was the product delivered on time?

# In[99]:


sns.barplot(x='Customer_rating', y='Reached.on.Time_Y.N', data=df)
plt.title('Customer Rating vs On-Time Delivery Rate')
plt.xlabel('Customer Rating (1=worst, 5=best)')
plt.ylabel('Proportion Delivered On Time')
plt.show()


# Q7: Is Customer Query Being Answered?

# In[93]:


#More calls → potential delivery issues or product dissatisfaction.
sns.boxplot(x='Reached.on.Time_Y.N', y='Customer_care_calls', data=df)
plt.title('Customer Care Calls vs Delivery Status')
plt.xlabel('Delivery On Time (1=Yes, 0=No)')
plt.ylabel('Number of Customer Care Calls')
plt.show()


# Q8: If Product Importance is High, does it have Highest Ratings or Get Delivered On Time?

# a. Product Importance vs On-Time Delivery:

# In[100]:


sns.barplot(x='Product_importance', y='Reached.on.Time_Y.N', data=df)
plt.title('Product Importance vs On-Time Delivery Rate')
plt.xlabel('Product Importance')
plt.ylabel('Proportion Delivered On Time')
plt.show()


# b. Product Importance vs Customer Rating:

# In[101]:


sns.boxplot(x='Product_importance', y='Customer_rating', data=df)
plt.title('Customer Rating by Product Importance')
plt.xlabel('Product Importance')
plt.ylabel('Customer Rating')
plt.show()


# In[ ]:




