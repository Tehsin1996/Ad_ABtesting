# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:56:53 2021

@author: a0981
"""


import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats


df = pd.read_csv('AdSmartABdata.csv')
print(df.head(5))

df.info()
df['browser'] = df['browser'].astype('str')

df.auction_id.nunique() == df.shape[0]
#Every auction id is unique

#check the distribution
sns.countplot(x="browser", data=df, hue="experiment")

df['ans'] = np.where(df.yes + df.no > 0, 1, 0)

# The % of users that respond to the survBIO questionnaireey
df['ans'].sum() / df['ans'].count()

# The % of users that respond to the survBIO questionnaireey for each AB-group
conv = df.groupby('experiment')['ans'].sum() / df.groupby('experiment')['ans'].count()
print(conv)

#Size for each group
size = df.groupby('experiment')['ans'].count()
print(size)

# Creating a list with bootstrapped means for each AB-group
boot_ans = []
for i in range(10000):
    boot_mean = df.sample(frac=1, replace=True).groupby('experiment')['ans'].mean()
    boot_ans.append(boot_mean)
    
# Transforming the list to a DataFrame
boot_ans = pd.DataFrame(boot_ans)

# Adding a column with the % difference between the two AB-groups
boot_ans['diff'] = (boot_ans['control'] - boot_ans['exposed']) /  boot_ans['control'] * 100

# Ploting the bootstrap % difference
ax = boot_ans['diff'].plot(kind = 'kde')
ax.set_xlabel("% difference in means")

# Calculating the probability that respond rate is greater in control group
prob = len(boot_ans[boot_ans['diff'] > 0]) / len(boot_ans)

# Pretty printing the probability
'{:.1%}'.format(prob)

print('Conclusion: If we want to increase respond rate we should show a creative, an online interactive ad, with the SmartAd brand to our customer.')

### Get P-value 
def get_pvalue(con_conv, test_conv, con_size, test_size):
    lift = -abs(test_conv - con_conv)
    
    scale_one = con_conv * (1-con_conv) * (1/ con_size)
    scale_two = test_conv * (1-test_conv) * (1/ test_size)
    scale_val = (scale_one + scale_two) ** 0.5
    
    p_value = 2 * stats.norm.cdf(lift, loc=0, scale = scale_val)  
    return p_value

con_conv = conv[0]
test_conv = conv[1]
con_size = size[0]
test_size = size[1]

p_value = get_pvalue(con_conv, test_conv, con_size, test_size) 

# Check for statistical significance
if p_value >= 0.05:
    print("Not Significant")
else:
    print("Significant Result")



