#%%
# this python script produces the results in the paper (the statistics, csv for figures)

# import libraries
import pandas as pd
import ast
from collections import Counter
# suppress warnings
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import cohen_kappa_score

# specify the label we will use (gpt-4o or gpt-4o-mini)
model = 'category_gpt-4o'

#%%
# import data
# main bot data
df_bots = pd.read_csv('data/bot_data.csv')
# bot labels 
df_bot_labels = pd.read_csv('data/bot_labels.csv')
# bot channel info
df_bot_channel_info = pd.read_csv('data/bot_channel_info.csv')
# bot translation
df_bot_translation = pd.read_csv('data/bot_translation.csv')
# merge the dataframes
df_bots = df_bots.merge(df_bot_labels, on='id_hash', how='left')
df_bots = df_bots.merge(df_bot_channel_info, on='id_hash', how='left')
df_bots = df_bots.merge(df_bot_translation, on='id_hash', how='left')

# convert command_list_en from string to list
df_bots['command_list_en'] = df_bots['command_list_en'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
df_bots['channels'] = df_bots['channels'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)

print('Disclaimer: this dataset does not contain messages that mention bots that might contain sensitive data.')
print("So you cannot reproduce the same labels (by humans & LLMs) because message was used for annotation.")

#%%
# Section 3 (Only bot data)
print('########### Section 3 ###########')
print('We have {} bots in total.'.format(df_bots.shape[0]))
print('We have {} bots with non-empty description'.format(df_bots[~df_bots['description'].isnull()].shape[0]))
print('We have {} bots with start response.'.format(df_bots[~df_bots['start_response'].isnull()].shape[0]))
print('We have {} bots with help response.'.format(df_bots[~df_bots['help_response'].isnull()].shape[0]))
print('We have {} bots with either start or help response.'.format(df_bots[~(df_bots['start_response'].isnull()) | ~(df_bots['help_response'].isnull())].shape[0]))

all_commands = []
bot_with_command = 0
for commands_list in df_bots['command_list_en'].dropna():
    commands = [i['command'] for i in commands_list if 'command' in i]
    if len(commands) > 0:
        bot_with_command += 1
    all_commands.extend(commands)

command_counts = Counter(all_commands)
print('There are {} bots with command list.'.format(bot_with_command))
print('The most common commands are:')
for command, count in command_counts.most_common(5):
    print(f"{command}: {round(100*count/bot_with_command, 2)}%")

#%%
# Section 4
print('########### Section 4 ###########')
categories = ["Admin Tools",
              "Content & Media",
              "Ideology",
              "Finance", 
              "Shopping",
              "Social & Gaming",
              "Underground",
              "Utility",
              "Others"
              ]

##  Section 4.1 (Domains)
# import the annotated samples
df_bots_annotated = pd.read_csv('data/bot_annotated_samples.csv')
# check the agreement between human annotators and AI models
def compute_agreement(annotation1, annotation2, categories=categories):
    agreement = (annotation1 == annotation2).mean()
    cohen_kappa = cohen_kappa_score(annotation1, annotation2, labels=categories)
    return agreement, cohen_kappa

h12_agree, h12_kappa = compute_agreement(df_bots_annotated['category_annotator_1'], df_bots_annotated['category_annotator_2'])
hg_agree, hg_kappa = compute_agreement(df_bots_annotated['category_human'], df_bots_annotated['category_gpt-4o'])
hgm_agree, hgm_kappa = compute_agreement(df_bots_annotated['category_human'], df_bots_annotated['category_gpt-4o-mini'])
print('Cohen kapp between annotator 1 and 2: {}, agreement: {}'.format(round(h12_kappa, 2), round(h12_agree, 2)))
print('Cohen kapp between human and {}: {}, agreement: {}'.format('gpt-4o', round(hg_kappa, 2), round(hg_agree, 2)))
print('Cohen kapp between human and {}: {}, agreement: {}'.format('gpt-4o-mini', round(hgm_kappa, 2), round(hgm_agree, 2)))

# create a table to compare the category distribution of two human annotators, gpt-4o and gpt-4o-mini
category_comparison = pd.DataFrame(index=categories, columns=['annotator_1', 'annotator_2', 'gpt-4o', 'gpt-4o-mini'])
for cat in categories:
    category_comparison.loc[cat, 'annotator_1'] = (df_bots_annotated['category_annotator_1'] == cat).mean()
    category_comparison.loc[cat, 'annotator_2'] = (df_bots_annotated['category_annotator_2'] == cat).mean()
    category_comparison.loc[cat, 'gpt-4o'] = (df_bots_annotated['category_gpt-4o'] == cat).mean()
    category_comparison.loc[cat, 'gpt-4o-mini'] = (df_bots_annotated['category_gpt-4o-mini'] == cat).mean()
print('Category distribution comparison:')
print(category_comparison)

#%%
# make sure all samples belong to the specified categories, otherwise put it as others
print('Model {}: {} ({}%) samples are not in the specified categories'.format('category_gpt-4o-mini', df_bots[~df_bots['category_gpt-4o-mini'].isin(categories)].shape[0], round(100*df_bots[~df_bots['category_gpt-4o-mini'].isin(categories)].shape[0]/df_bots.shape[0], 3)))
print('Model {}: {} ({}%) samples are not in the specified categories'.format('category_gpt-4o', df_bots[~df_bots['category_gpt-4o'].isin(categories)].shape[0], round(100*df_bots[~df_bots['category_gpt-4o'].isin(categories)].shape[0]/df_bots.shape[0], 3)))

# put samples not in the specified categories as 'Others'
df_bots['category_gpt-4o-mini'] = df_bots['category_gpt-4o-mini'].apply(lambda x: x if x in categories else 'Others')
df_bots['category_gpt-4o'] = df_bots['category_gpt-4o'].apply(lambda x: x if x in categories else 'Others')

#%%
# category distribution by the number of bots
print('Category distribution:')
category_dist = df_bots[model].value_counts(normalize=False).sort_values(ascending=False)
category_dist.name = 'num_bots'

# get the number of unique channels for each category
category2channels = {}
for cat in categories:
    channels = set()
    for channels_list in df_bots[df_bots[model] == cat]['channels']:
        channels.update(channels_list)
    category2channels[cat] = len(channels)
# covert to df
category_dist_channels = pd.Series(category2channels).sort_values(ascending=False)
# set name as 'num_channels'
category_dist_channels.name = 'num_channels'

# concatenate two series
df_bots_agg = pd.concat([category_dist, category_dist_channels], axis=1)
df_bots_agg['percentage_bots'] = 100*df_bots_agg['num_bots']/df_bots_agg['num_bots'].sum()

# print the percentage of each domain
print(df_bots_agg[['percentage_bots']])

# save data for figure 
df_bots_agg.to_csv('csv/bot_category.csv')
print('Saved bot category distribution as csv/bot_category.csv')

#%%
## Section 4.2 (Functionalities)
# identify the functionality based on the command list
payment_keyword = ['pay', 'payment', 'purchase', 'buy', 'sell', 'deposit', 'withdraw', 'setwallet']  # 'donate', 'balance'
referral_keyword = ['refer', 'referral', 'invite']  # , 'join']
referral_keyword_exclude = ['reference']
crowdsource_keyword = ['upload', 'submit', 'report']
crowdsource_keyword_exclude = ['reports']

# extract the command:description of matched commands 
def extract_matched_commands(command_list, keywords, keywords_exclude=[]):
    matched_commands = []
    if isinstance(command_list, list):
        for command in command_list:
            if ('command' in command) and (any(keyword in command['command'].lower() for keyword in keywords)) and not (any(exclude in command['command'].lower() for exclude in (keywords_exclude))):
                matched_commands.append(command)
    return matched_commands

# extract the command for each bot
# payment 
df_bots['matched_payment_commands'] = df_bots['command_list_en'].apply(lambda x: extract_matched_commands(x, payment_keyword))
df_bots['contains_payment_command'] = df_bots['matched_payment_commands'].apply(lambda x: len(x)>0)
print('Payment bot count (matched): {} ({}%)'.format(df_bots['contains_payment_command'].sum(), round(100*df_bots['contains_payment_command'].sum()/bot_with_command, 2)))
all_matched_payment_commands = []
for commands in df_bots['matched_payment_commands']:
    all_matched_payment_commands.extend(commands)

payment_command_counts = Counter([cmd['command'] for cmd in all_matched_payment_commands])
# bundle the related command together (manually look at command with at least 2 counts)
payment_command_bundle = {'withdraw': ['withdraw', 'withdrawal', 'quickwithdraw', 'withdrawals'], \
    'wallet': ['setwallet'], \
    'deposit': ['deposit', 'deposits'], \
    'pay': ['pay', 'payment', 'paysupport', 'paypal', 'spay', 'payout', 'payouts', 'paytoken'],
    'buy': ['buy', 'purchase', 'buysell', 'buyminer', 'buy_token', 'buy_all', 'buy_ads', 'purchases', 'buyorrent']}

payment_command_bundle_counts = {}
for bundle_name, bundle_keywords in payment_command_bundle.items():
    count = sum([payment_command_counts[keyword] for keyword in bundle_keywords])
    payment_command_bundle_counts[bundle_name] = count
print('The most common matched payment command bundles are:')
print(sorted(payment_command_bundle_counts.items(), key=lambda x: x[1], reverse=True))

# referral
df_bots['matched_referral_commands'] = df_bots['command_list_en'].apply(lambda x: extract_matched_commands(x, referral_keyword, referral_keyword_exclude))
df_bots['contains_referral_command'] = df_bots['matched_referral_commands'].apply(lambda x: len(x)>0)
print('Referral bot count (matched): {} ({}%)'.format(df_bots['contains_referral_command'].sum(), round(100*df_bots['contains_referral_command'].sum()/bot_with_command, 2)))
all_matched_referral_commands = []
for commands in df_bots['matched_referral_commands']:
    all_matched_referral_commands.extend(commands)

referral_command_counts = Counter([cmd['command'] for cmd in all_matched_referral_commands])
referral_command_bundle = {'referral': ['referral', 'referrals', 'refer', 'referals', 'referral_shop', 'referrallink', 'referralcommand', 'myreferrals'], \
    'invite': ['invite', 'invite_friends', 'newinvite', 'invitefriends', 'invites', 'invitelink', 'invite_link']}
print('The most common matched referral command bundles are:')
referral_command_bundle_counts = {}
for bundle_name, bundle_keywords in referral_command_bundle.items():
    count = sum([referral_command_counts[keyword] for keyword in bundle_keywords])
    referral_command_bundle_counts[bundle_name] = count
print(sorted(referral_command_bundle_counts.items(), key=lambda x: x[1], reverse=True))

# crowdsource
df_bots['matched_crowdsource_commands'] = df_bots['command_list_en'].apply(lambda x: extract_matched_commands(x, crowdsource_keyword, crowdsource_keyword_exclude))
df_bots['contains_crowdsource_command'] = df_bots['matched_crowdsource_commands'].apply(lambda x: len(x)>0)
print('Crowdsource bot count (matched): {} ({}%)'.format(df_bots['contains_crowdsource_command'].sum(), round(100*df_bots['contains_crowdsource_command'].sum()/bot_with_command, 2)))
all_matched_crowdsource_commands = []
for commands in df_bots['matched_crowdsource_commands']:
    all_matched_crowdsource_commands.extend(commands)

crowdsource_command_counts = Counter([cmd['command'] for cmd in all_matched_crowdsource_commands])
crowdsource_command_bundle = {'report': ['report', 'bugreport'], \
    'submit': ['submit'], \
    'upload': ['upload', 'photos_uploaded', 'uploadlang']}

print('The most common matched crowdsource command bundles are:')
crowdsource_command_bundle_counts = {}
for bundle_name, bundle_keywords in crowdsource_command_bundle.items():
    count = sum([crowdsource_command_counts[keyword] for keyword in bundle_keywords])
    crowdsource_command_bundle_counts[bundle_name] = count
print(sorted(crowdsource_command_bundle_counts.items(), key=lambda x: x[1], reverse=True))

# %%
# check the description contains AI related keywords 
from keyword_matching import contains_keywords

# Exact word matches (case-insensitive, word boundaries) partly produced by Claude
ai_keywords = [ 
    'ai', 'artificial-intelligence', 'machine-learning', 'ml',
    'neural-network', 'deep-learning', 'llm', 'large-language-model',
    'gpt', 'chatgpt', 'claude', 'bard', 'gemini',
    'ocr', 'image-generation', 'text-generation', 'translation',
    'midjourney', 'stable-diffusion', 'deepfake', 'deepfakes',
    'bert', 'roberta', 'embeddings',
    'nlp', 'natural-language-processing', 'computer-vision',
    'reinforcement-learning', 'supervised-learning', 'unsupervised-learning',
    'generative-ai', 'artificial-neural-network', 'convolutional-neural-network',
    'recurrent-neural-network', 'rnn', 'lstm', 'gru', 'autoencoder',
    'dall-e', 'gpt-3', 'gpt-4',
    'ai-powered', 'ai-generated', 'ai-driven',
]
print('We have {} AI related keywords.'.format(len(ai_keywords)))

matched_ai_keywords = []
for desc in df_bots['description_en']:
    if type(desc) == str:
        matched_ai_keywords.append(contains_keywords(desc, ai_keywords))
    else:
        matched_ai_keywords.append(list())

# the keyword matching might count twice (e.g., AI powerd --> AI, AI-powered)
df_bots['matched_ai_keywords'] = matched_ai_keywords
df_bots['contains_ai_keyword'] = df_bots['matched_ai_keywords'].apply(lambda x: len(x)>0)
print('AI related bot count: {} ({}%)'.format(df_bots['contains_ai_keyword'].sum(), round(100*df_bots['contains_ai_keyword'].sum()/df_bots.shape[0], 2)))

# get the most common keywords
all_matched_ai_keywords = []
for keywords in df_bots['matched_ai_keywords']:
    all_matched_ai_keywords.extend(list(keywords))
ai_keyword_counts = Counter(all_matched_ai_keywords)
print('The most common AI related keywords in bot descriptions are:')
print(ai_keyword_counts.most_common(10))

#%%
# Section 4.3 (cateogry vs functinality)
# create a table where y-axis is the funcitonality and x-axis is the category
functionality_cols = ['contains_payment_command', 'contains_referral_command', 'contains_crowdsource_command', 'contains_ai_keyword']
# number of commands 
df_bots['num_command_list'] = df_bots['command_list_en'].apply(lambda x: len(x) if isinstance(x, list) else 0)

df_functionality_category = pd.DataFrame(index=functionality_cols, columns=categories)
for func in functionality_cols:
    for cat in categories:
        count = df_bots[(df_bots[func]) & (df_bots[model] == cat)].shape[0]
        df_functionality_category.loc[func, cat] = count

# convert to int
df_functionality_category = df_functionality_category.astype(int)
# normalize by column (by the total number of bots in each category)
df_functionality_category_normalized = df_functionality_category.div(df_bots[model].value_counts(), axis=1)
# for payment, referral, crowdsource, only consider bots with at least one command
df_functionality_category_normalized.loc['contains_payment_command'] = df_functionality_category.loc['contains_payment_command'] / df_bots[df_bots['num_command_list']>0][model].value_counts()
df_functionality_category_normalized.loc['contains_referral_command'] = df_functionality_category.loc['contains_referral_command'] / df_bots[df_bots['num_command_list']>0][model].value_counts()
df_functionality_category_normalized.loc['contains_crowdsource_command'] = df_functionality_category.loc['contains_crowdsource_command'] / df_bots[df_bots['num_command_list']>0][model].value_counts()
# I was going to normalize AI related bots by the number of bots with description, but I decided not to do so
# df_functionality_category_normalized.loc['contains_ai_keyword'] = df_functionality_category.loc['contains_ai_keyword'] / df_bots[df_bots['num_command_list']>0][model].value_counts()

# export to csv
df_functionality_category_normalized.to_csv('csv/functionality_category_normalized.csv')
print('Saved functionality vs category as csv/functionality_category_normalized.csv')

#%%
## Section 4.4 (malicious bots)
# bot with warning 
print('Bot with warning count:', df_bots['warning'].sum())
df_bots_warning = df_bots[df_bots['warning']==True]
df_bots_warning['command_list_len'] = df_bots_warning['command_list_en'].apply(lambda x: len(x) if isinstance(x, list) else 0)
# finance warning bot, look at functionality (%)
df_bots_finance_warning = df_bots_warning[(df_bots_warning[model]=='Finance')]
print('There are {} finance bots with warning.'.format(df_bots_finance_warning.shape[0]))
for functionality in ['contains_payment_command', 'contains_referral_command', 'contains_crowdsource_command']:
    print('{}: {}%'.format(functionality, round(100*df_bots_finance_warning[df_bots_finance_warning[functionality]].shape[0]/df_bots_finance_warning[(df_bots_finance_warning['command_list_len']>0)].shape[0], 2)))

# %%
# Section 5 
print('########### Section 5 ###########')
# Section 5.1 (Language) 
print('Language distribution in bot description:')
language_col = 'description_language'
language_col2 = 'message_most_frequent_language'
lang_code_map = {'en': 'English', 'ru': 'Russian', 'fa': 'Farsi', 'ar': 'Arabic', 'es': 'Spanish', 'uk': 'Ukrainian'}
# print like English X%, Russian Y%, ... (not in pandas)
print(df_bots[language_col].value_counts(normalize=True).iloc[:5].rename(index=lang_code_map).to_string())
print('Language distribution in messages that mention bots:')
print(df_bots[language_col2].value_counts(normalize=True).iloc[:5].rename(index=lang_code_map).to_string())

#%%
# create a table where y-axis is the language and x-axis is the category
top_n_languages = 5
top_languages = list(df_bots[language_col2].value_counts().nlargest(top_n_languages).index)
df_language_category = pd.DataFrame(index=top_languages, columns=categories)
for lang in top_languages:
    for cat in categories:
        count = df_bots[(df_bots[language_col2] == lang) & (df_bots[model] == cat)].shape[0]
        df_language_category.loc[lang, cat] = count
# convert to int
df_language_category = df_language_category.astype(int)
# normalize by column (by the total number of bots in each category)
df_language_category_normalized = df_language_category.div(df_bots[model].value_counts(), axis=1)

# export into csv
df_language_category_normalized.to_csv('csv/language_category_normalized.csv')
print('Saved language vs category as csv/language_category_normalized.csv')

#%%
## Section 5.2 (Usage)
# look at the life span across categories
# overall lifespan days distribution (min, 25%, 50%, 75%, max, avg)
df_bots['lifespan_days'] = (df_bots['last_timestamp'] - df_bots['first_timestamp'])/(60*60*24)
print('Median and average lifespan days overall is {} and {}, respectively.'.format(round(df_bots[df_bots['num_msg']>1]['lifespan_days'].median(), 2), round(df_bots[df_bots['num_msg']>1]['lifespan_days'].mean(), 2)))

# median lifespan days for each category
print('Median lifespan days for each category:')
print(df_bots[df_bots['num_msg']>1].groupby(model)['lifespan_days'].median().to_string())

# save the data for boxplot (exclude bots with 1 or less message or Others category)
df_bots_lifespan_days = df_bots[(df_bots['num_msg']>1)&(df_bots[model]!='Others')][[model, 'lifespan_days']]
df_bots_lifespan_days.to_csv('csv/bot_lifespan_days.csv', index=False)
print('Saved bot lifespan days as csv/bot_lifespan_days.csv')

# %%
# Reuse of bots
# count the number of duplicated descriptions for each category
# remove NaN descriptions (empty description)
df_bots_with_description = df_bots.dropna(subset=['description'])
# make sure the description has certain length
# the threshold is set to 10 (by manually look at short descriptions e.g., Hi, Hello, . , -)
df_bots_with_description = df_bots_with_description[df_bots_with_description['description'].apply(lambda x: len(x)>=10)]
print('There are {} bots with (non-short) description ({}%).'.format(df_bots_with_description.shape[0], round(100*df_bots_with_description.shape[0]/df_bots.shape[0], 2)))
# reorder by id
df_bots_with_description = df_bots_with_description.sort_values('id_hash')
# get the duplicated descriptions
df_description_counts = df_bots_with_description.groupby(['description']).agg(
    count=('id_hash', 'count'),
    username=('name_hash', lambda x: list(x)),
    description_en = ('description_en', 'first'),
    # count the number of bots per category (sort by the count)
    categories=(model, lambda x: Counter(x).most_common()),
    warning=('warning', 'sum')
).reset_index() 

# only keep descriptions that are duplicated (count > 1) including the original bot
df_description_counts = df_description_counts[df_description_counts['count']>1]
print('There are {} duplicated descriptions ({}%)'.format(df_description_counts['count'].sum(), round(100*df_description_counts['count'].sum()/df_bots.shape[0], 2)))
df_description_counts.sort_values('count', ascending=False, inplace=True)
print('Max reuse count:', df_description_counts['count'].max())

# count the number of duplicated bots per cateogry 
category_count_agg = {i:0 for i in categories}
for category_count in df_description_counts['categories']:
    for cat, count in category_count:
        category_count_agg[cat] += count
category_count_agg = {k: v for k, v in sorted(category_count_agg.items(), key=lambda item: item[1], reverse=True)}
print('Finance account for {} ({}%) of duplicated bots.'.format(category_count_agg['Finance'], round(100*category_count_agg['Finance']/df_description_counts['count'].sum(), 2)))
assert sum(category_count_agg.values()) == df_description_counts['count'].sum()
print('This dataset does not have the original username, so we do not calculate the similarity of usernames.')

#%%
# overall trend
# how many bots were active at each month (time-series)
# convert to unix timestamp 
df_bots['first_timestamp_date'] = pd.to_datetime(df_bots['first_timestamp'], unit='s')
df_bots['last_timestamp_date'] = pd.to_datetime(df_bots['last_timestamp'], unit='s')
df_bots['first_timestamp_month'] = df_bots['first_timestamp_date'].dt.to_period('M')
df_bots['last_timestamp_month'] = df_bots['last_timestamp_date'].dt.to_period('M')
# create a time series from the earliest month to the latest month
time_series = pd.period_range(start=df_bots['first_timestamp_month'].min(), end=df_bots['last_timestamp_month'].max(), freq='M')
active_bots_per_month = []
# remove the final month since it is less complete
for month in time_series[:-1]:
    count = df_bots[(df_bots['first_timestamp_month'] <= month) & (df_bots['last_timestamp_month'] >= month)].shape[0]
    count_ai = df_bots[(df_bots['contains_ai_keyword']) & (df_bots['first_timestamp_month'] <= month) & (df_bots['last_timestamp_month'] >= month)].shape[0]
    active_bots_per_month.append({'month': month.to_timestamp(), 'active_bots': count, 'active_ai_bots_ratio': 100*count_ai/count})
df_active_bots_per_month = pd.DataFrame(active_bots_per_month)

# save it for fig
df_active_bots_per_month.to_csv('csv/active_bots_per_month.csv', index=False)
print('Saved active bots per month as csv/active_bots_per_month.csv')

# %%
# Section 5.3 (Topology)
n = 4
df_bots['num_channels'] = df_bots['channels'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df_bots_network = df_bots[(df_bots['num_channels']>=n)].melt(id_vars=[model], 
                   value_vars=['avg_degree', 'density'],
                   var_name='variable', value_name='value')

# drop rows with NaN value
df_bots_network = df_bots_network.dropna(subset=['value'])
# rename avg_degree to Average Degree, density to Density
df_bots_network['variable'] = df_bots_network['variable'].replace({'avg_degree': 'Average Degree', 'density': 'Density'})
df_bots_network.to_csv('csv/bot_network_metrics.csv', index=False)
print('Saved bot network metrics as csv/bot_network_metrics.csv')
print('Note: the example channel networks are not produced in this script because it requires additional context about the channels.')