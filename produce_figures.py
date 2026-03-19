# this script produces figures used in the paper
#%%
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# suppress warnings
import warnings
warnings.filterwarnings('ignore')

# specify the model label used for category
model = 'category_gpt-4o'
categories = ['Admin Tools', 'Content & Media', 'Finance', 'Ideology', 'Shopping', 'Social & Gaming', 'Underground', 'Utility']

#%%
# get the number of bots per category 
df_bots_agg = pd.read_csv('csv/bot_category.csv', index_col=0)
# visualize the number of bots and channels for each category in a bar plot
# x-axis is category, y-axis is number of bots and number of channels
# use matplotlib and seaborn
# exclude 'Others' category
df_bots_agg = df_bots_agg[df_bots_agg.index!='Others']
# order by category variable
df_bots_agg = df_bots_agg.reindex(categories)

# visualize
plt.figure(figsize=(10, 3))
bar_width = 0.4
index = np.arange(len(df_bots_agg))
bar1 = plt.bar(index, df_bots_agg['num_bots'], bar_width, label='Nr of Bots', color='b')
bar2 = plt.bar(index + bar_width, df_bots_agg['num_channels'], bar_width, label='Nr of Channels', color='r')
# change x-axis tick labels to the short names
xaxis_labels = {'Admin Tools': 'AT', 'Content & Media': 'CM', 'Finance': 'FN', 'Ideology': "ID", 'Shopping': 'SP', 'Social & Gaming':'SG', 'Underground':'UG', 'Utility': 'UT'}
plt.xticks(index + bar_width / 2, [xaxis_labels[i] for i in categories], fontsize=20)
plt.yticks(fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.xlabel('')
plt.legend(fontsize=16)
plt.tight_layout()
# save the figure
plt.savefig('figs/num_bots_channels_per_category.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
# category vs functionality 
df_functionality_category_normalized = pd.read_csv('csv/functionality_category_normalized.csv', index_col=0)
# exclude 'Others' column
df_functionality_category_normalized = df_functionality_category_normalized.drop(columns=['Others'])

# visualize
plt.figure(figsize=(10, 2.5))
# heatmap with annot
sns.heatmap(df_functionality_category_normalized, annot=True, fmt=".2f", cmap="YlGnBu", annot_kws={"size": 12})
plt.title('')
plt.ylabel('')
# change names in yaxis label 
yaxis_labels = {'contains_payment_command': 'Payment', 'contains_referral_command': 'Referral', 'contains_crowdsource_command': 'Crowdsource', 'contains_ai_keyword': 'AI'}
functionality_cols = yaxis_labels.keys() 
plt.yticks(ticks=np.arange(len(functionality_cols))+0.5, labels=[yaxis_labels[i] for i in functionality_cols], rotation=0, fontsize=16)
plt.xlabel('Category')
# change the label in x-axis
xaxis_labels = {'Admin Tools': 'AT', 'Content & Media': 'CM', 'Finance': 'FN', 'Ideology': "ID", 'Shopping': 'SP', 'Social & Gaming':'SG', 'Underground':'UG', 'Utility': 'UT'}
category_cols = xaxis_labels.keys()
plt.xticks(ticks=np.arange(len(category_cols))+0.5, labels=[xaxis_labels[i] for i in category_cols], rotation=0, fontsize=16)
plt.xlabel('')
# save the figure
plt.tight_layout()
plt.savefig('figs/functionality_category_heatmap.pdf', dpi=300)
plt.show()

# %%
# do the same for language vs category
df_language_category_normalized = pd.read_csv('csv/language_category_normalized.csv', index_col=0)
# exclude 'Others' column
df_language_category_normalized = df_language_category_normalized.drop(columns=['Others'])

# visualize
plt.figure(figsize=(10, 3))
# heatmap with annot
sns.heatmap(df_language_category_normalized, annot=True, fmt=".2f", cmap="YlGnBu", annot_kws={"size": 12})
plt.title('')
plt.ylabel('')
yaxis_labels = {'ru': 'Russian', 'en': 'English', 'fa': 'Farsi', 'ar': 'Arabic', 'es': 'Spanish'}
# reorder by the original order
yaxis_labels = {k: yaxis_labels[k] for k in df_language_category_normalized.index}
top_languages = yaxis_labels.keys()
plt.yticks(ticks=np.arange(len(top_languages))+0.5, labels=[yaxis_labels[i] for i in top_languages], rotation=0, fontsize=16)
plt.xlabel('')
category_cols = xaxis_labels.keys()
plt.xticks(ticks=np.arange(len(category_cols))+0.5, labels=[xaxis_labels[i] for i in category_cols], rotation=0, fontsize=16)
# no xaxis label
plt.xlabel('')
# save the figure
plt.tight_layout()
plt.savefig('figs/language_category_heatmap.pdf', dpi=300)
plt.show()

# %%
# lifespan days vs category
df_bots_lifespan_days = pd.read_csv('csv/bot_lifespan_days.csv')
# order the category 
df_bots_lifespan_days[model] = pd.Categorical(df_bots_lifespan_days[model], categories=categories, ordered=True)
# boxplot of lifespan_days for each category
# fig size
plt.figure(figsize=(10, 4))
sns.boxplot(x=model, y='lifespan_days', data=df_bots_lifespan_days)
plt.yscale('log')
plt.title('')
# font size of y label
plt.ylabel('Lifespan Days (log scale)', fontsize=18)
plt.xlabel('Category', fontsize=18)
plt.xticks(ticks=np.arange(len(categories)), labels=[xaxis_labels[i] for i in categories], rotation=0, fontsize=20)
# y-axis 10, 100, 1000 instead of 1e1, 1e2, 1e3
plt.yticks(ticks=[10, 100, 1000], labels=['10', '100', '1,000'], fontsize=18)
plt.tight_layout()
plt.savefig('figs/bot_lifespan_days_by_category.pdf', dpi=300)
# save the figure
plt.savefig('figs/active_bots_over_time.pdf', dpi=300)
plt.show()

# %%
# Nr of bots overtime
df_active_bots_per_month = pd.read_csv('csv/active_bots_per_month.csv')
# convert month to datetime
df_active_bots_per_month['month'] = pd.to_datetime(df_active_bots_per_month['month'])

# visualize
fig, ax1 = plt.subplots(figsize=(10, 2.5))
ax1.plot(df_active_bots_per_month['month'], df_active_bots_per_month['active_bots'], label='Active Bots')
ax1.set_ylabel('Nr of Active Bots', fontsize=14)
ax1.set_ylim(0, max(df_active_bots_per_month['active_bots'])*1.1)
# tick font size
ax1.tick_params(axis='y', labelsize=14)
# ai bot ratio on secondary y-axis
ax2 = ax1.twinx()
ax2.plot(df_active_bots_per_month['month'], df_active_bots_per_month['active_ai_bots_ratio'], color='orange', label='AI Bot Ratio')
ax2.set_ylabel('AI Bot Ratio (in %)', fontsize=14)
ax2.tick_params(axis='y', labelsize=14)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, shadow=False, fontsize=16)
ax1.tick_params(axis='x', labelsize=16)
plt.tight_layout()
plt.title('')
# save the figure
plt.savefig('figs/active_bots_over_time.pdf', dpi=300)
plt.show()

# %%
# network metrics for each bot
df_bots_network = pd.read_csv('csv/bot_network_metrics.csv')
# exclude Others category
df_bots_network = df_bots_network[df_bots_network[model]!='Others']
# seperate the data
avg_degree_data = df_bots_network[df_bots_network['variable'] == 'Average Degree']
density_data = df_bots_network[df_bots_network['variable'] == 'Density']
# order the category 
avg_degree_data[model] = pd.Categorical(avg_degree_data[model], categories=categories, ordered=True)
density_data[model] = pd.Categorical(density_data[model], categories=categories, ordered=True)

fig, ax1 = plt.subplots(figsize=(12, 4))

sns.boxplot(data=avg_degree_data, x=model, y='value', ax=ax1, 
            color='lightblue', width=0.3)
ax1.set_ylabel('Average Degree', color='blue', fontsize=16)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.tick_params(axis='y', labelsize=16)
ax1.set_ylim(0, 3)

# Secondary y-axis for Density
ax2 = ax1.twinx()

# create offset positions for density boxes
positions = [i + 0.3 for i in range(len(categories))]

# create offset box plots for density
for i, cat in enumerate(categories):
    density_values = density_data[density_data[model] == cat]['value']
    bp = ax2.boxplot(density_values, positions=[positions[i]], widths=0.3, 
                    # fill the box with color 
                    patch_artist=True, 
                    # 70% transparent
                    boxprops=dict(facecolor='lightcoral', alpha=0.7),
                    # box line (median, whisker, cap)
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='red'),
                    capprops=dict(color='red'),
                    # outlier 
                    flierprops=dict(markerfacecolor='red', marker='o'))

ax2.set_ylabel('Density', color='red', fontsize=16)
ax2.tick_params(axis='y', labelcolor='red')
# tick font size
ax2.tick_params(axis='y', labelsize=16)
ax2.set_ylim(0, 0.8)

# Set x-axis with proper centering
ax1.set_xticks([i + 0.15 for i in range(len(categories))])
ax1.set_xticklabels([xaxis_labels[i] for i in categories], rotation=0, fontsize=18)
# no xaxis label
ax1.set_xlabel('') 
plt.title('')
plt.tight_layout()
# save the figure
plt.savefig('figs/bot_network_metrics_by_category.pdf', dpi=300)
print('Successfully produced all figures so check figs/ directory!')
plt.show()
