#!/usr/bin/env python
# coding: utf-8

# # Hackerranck Project

# In[1]:


from csv import reader
opened_file = open('hacker_news.csv')
read_file = reader(opened_file)
hn = list(read_file)


# In[3]:


print(hn[:5])


# In[4]:


headers = hn[0]
hn = hn[1:]
print(hn[:5])


# In[5]:


show_posts = []
other_posts = []
ask_posts = []

for row in hn:
    title = row[1]
    
    if title.lower().startswith('ask hn'):
        ask_posts.append(row)
        
    elif title.lower().startswith('show hn'):
        show_posts.append(row)
    else:
        other_posts.append(row)
print(len(ask_posts))
print(len(show_posts))
print(len(other_posts))


# In[6]:


print(ask_posts[:5])
print(show_posts[:5])



# In[7]:


total_ask_comments = 0
for each in ask_posts:
    n_of_comments = each[4]
    total_ask_comments += int(n_of_comments)
avg_ask_comments = total_ask_comments / len(ask_posts)
print(avg_ask_comments)


# In[8]:


total_show_comments = 0
for each in show_posts:
    n_of_comments = each[4]
    total_show_comments += int(n_of_comments)
avg_show_comments = total_show_comments / len(show_posts)
print(avg_show_comments)


# # Comparison
# **avg ask comments = 14.038 while**
# **avg show comments = 10.316**
# 

# In[9]:


# 7. sırada created at var ama ben bunu 6. indeksle okuyacağım 8/4/2016 11:52 bu tarz bir date ve time 
import datetime as dt 

result_list = []

for element in ask_posts:
    created_at =element[6] # created_at
    number_of_comments = element[4] # number_of_comments
    result_list.append([created_at,number_of_comments])
    
print(result_list[:3])    
    


# In[10]:


dummy = '8/16/2016 9:55'
date_obj = dt.datetime.strptime(dummy,'%m/%d/%Y %H:%M')
hour= dt.datetime.strftime(date_obj,'%H:%M')
print(date_obj)
print(hour)


# In[19]:


print(result_list[0])


# In[20]:


counts_by_hour ={}
comments_by_hour ={}
# extracting hour from date
for each in result_list:
    date_obj = dt.datetime.strptime(each[0],'%m/%d/%Y %H:%M')
    hour= dt.datetime.strftime(date_obj,'%H')
    number_of_comments = int(each[1])
    if hour not in counts_by_hour:
        counts_by_hour[hour]=1
        comments_by_hour[hour] = number_of_comments
        
    if hour in counts_by_hour:
        counts_by_hour[hour]+=1
        comments_by_hour[hour] += number_of_comments


# **counts_by_hour:** contains the number of ask posts created during each hour of the day
# **comments_by_hour:** contains the corresponding number of comments ask posts created at each hour received

# In[23]:


print(counts_by_hour)


# In[24]:


print(comments_by_hour)


# In[33]:


comment_list = []

for hour in comments_by_hour:
    comment_list.append([hour,comments_by_hour[hour],counts_by_hour[hour]])


# In[35]:


print(comment_list[2])


# In[39]:


avg_by_hour = [] 
for element in comment_list:
    avg = element[1] / element[2]
    avg_by_hour.append([element[0],avg])
print(avg_by_hour)


# In[42]:


swap_avg_by_hour = []

for row in avg_by_hour:
    first = row[1]
    second = row[0]
    swap_avg_by_hour.append([first,second])
    
sorted_swap = sorted(swap_avg_by_hour, reverse=True)
print(sorted_swap[:5])


# In[44]:


print( "Top 5 Hours for Ask Posts Comments")


# In[46]:


for each in sorted_swap[:5]:
    print('{}:00: {:.2f} average commments per post'.format(each[1],each[0]))

