import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('C:/Users/Daniel/Desktop/BGU/Year 5/Semester A/ML/project/ML2021/ML2021-benny/data.csv')
X_train = pd.read_csv('data.csv')
y_train = pd.read_csv('y_train.csv')
train_data = pd.concat([pd.read_csv('X_train.csv '), pd.read_csv('y_train.csv')], axis=1)
X_test = pd.read_csv('x_test.csv')
sns.set()

# count samples
print('count x_train {}'.format(len(X_train.index)))
print('count y_train {}'.format(len(y_train.index)))
print('count x_test {}'.format(len(X_test.index)))
print()

# count features
print('count features {}'.format(len(X_train.columns)))
print()

# get min and max year of release
print('info about X_train[Year_of_Release]')
print(X_train['Year_of_Release'].describe())
print()

# count max duplicated games(same entity, different sample)
dup_count = X_train.pivot_table(index=['Name'], aggfunc='size')
print(dup_count[dup_count == dup_count.max()])
print()

# count missing values
print('X_train missing values {}'.format(X_train.isnull().sum().sum()))
print('y_train missing values {}'.format(y_train.isnull().sum().sum()))
print('X_test missing values {}'.format(X_test.isnull().sum().sum()))
print()

# create histograms and density graphs
hist_list = ['Platform', 'Year_of_Release', 'Genre', 'Rating']
fig, axs = plt.subplots(2, 2)
fig.subplots_adjust(hspace=0.5, wspace=0.5)
fig.suptitle('Distributions of Features Part 1')
for ax, feature in zip(axs.flatten(), hist_list):
    ax.set(title=feature.upper())
    sns.countplot(x=feature, data=train_data, ax=ax)
    ax.get_xaxis().set_label_text('')
    ax.tick_params(axis='x', rotation=90)
plt.show()


density_list = [df.columns.drop('state')]
fig, axs = plt.subplots(7, 7)
fig.subplots_adjust(hspace=0.75, wspace=0.5)
fig.suptitle('Distributions of Features Part 2')
for ax, feature in zip(axs.flatten(), density_list):
    sns.distplot(df, ax=ax, hist=False, kde=True, bins=100, color='blue',
                 kde_kws={'linewidth': 1})
    ax.get_xaxis().set_label_text('{} in mill units'.format(feature))
plt.show()


critic_list = ['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']
fig, axs = plt.subplots(2, 2)
fig.subplots_adjust(hspace=0.5, wspace=0.5)
fig.suptitle('Distributions of Features Part 3')
for ax, feature in zip(axs.flatten(), critic_list):
    ax.set(title=feature.upper())
    sns.distplot(train_data[feature], ax=ax, hist=False, kde=True, bins=100, color='blue',
                 kde_kws={'linewidth': 1})
    ax.get_xaxis().set_label_text('')
plt.show()


hard_list = ['Name', 'Publisher', 'Developer', 'Reviewed']
fig, axs = plt.subplots(2, 2)
fig.subplots_adjust(hspace=0.5, wspace=0.5)
fig.suptitle('Distributions of Features Part 4')
for ax, feature in zip(axs.flatten()[:-1], hard_list[:-1]):
    ax.set(title=feature.upper())
    counts_list = train_data[feature].value_counts().tolist()
    sns.distplot(counts_list, ax=ax, hist=True, kde=False, bins=70, color='blue', )
    ax.get_xaxis().set_label_text('')

reviewed_count = pd.DataFrame({"answer": ["Yes", "No"],
                               "count": [len(train_data[train_data['Reviewed'] == 'YES'].index),
                                         len(train_data[train_data['Reviewed'] == 'NO'].index)]})
axs[1, 1].set(title='Reviewed')
axs[1, 1] = sns.barplot(x="answer", y="count", data=reviewed_count)
axs[1, 1].get_xaxis().set_label_text('')
plt.show()
hard_list = ['Name', 'Publisher', 'Developer', 'Reviewed']
fig, axs = plt.subplots(2, 2)
fig.subplots_adjust(hspace=1, wspace=0.5)
fig.suptitle('Distributions of Features Part 4')
for ax, feature in zip(axs.flatten()[:-1], hard_list[:-1]):
    ax.set(title=feature.upper())
    counts = pd.cut(train_data[feature].value_counts().reset_index()[feature], 15)
    sns.barplot(x=counts.values, y=counts.index, ax=ax, color='blue')
    ax.tick_params(labelrotation=90, labelsize=4)
    ax.set_xlabel('Repetitions from each sample', fontsize=10)
    ax.set_ylabel('Instances in each bin', fontsize=10)

reviewed_count = pd.DataFrame({"answer": ["Yes", "No"],
                               "count": [len(train_data[train_data['Reviewed'] == 'YES'].index),
                                         len(train_data[train_data['Reviewed'] == 'NO'].index)]})
axs[1, 1].set(title='Reviewed')
axs[1, 1] = sns.barplot(x="answer", y="count", data=reviewed_count)
axs[1, 1].get_xaxis().set_label_text('Answer')
axs[1, 1].get_yaxis().set_label_text('Count')
plt.show()

# print min & max values
for col in train_data.columns:
    if train_data[col].dtype == np.int64 or train_data[col].dtype == np.float64:
        print("{}: Min:{}, Max:{}".format(col, train_data[col].min(), train_data[col].max()))

# print density of target feature
fig = plt.figure()
sns.distplot(train_data['EU_Sales'], hist=False, kde=True, bins=100, color='blue', kde_kws={'linewidth': 1})
plt.title('Game sales in Europe')
plt.xlabel('Millions of units')
plt.ylabel('Density')
plt.show()

# A Scatter plot of the Critic Score and the Total sales of the games
sum_list = ['NA_Sales', 'JP_Sales', 'Other_Sales', 'EU_Sales']
train_data['Total_Sales'] = train_data[sum_list].sum(axis=1)
sns.scatterplot(x='Critic_Score', y='Total_Sales', data=train_data)
plt.title('Critic Score and the Total sales')
plt.xlabel('Critic Score')
plt.ylabel('Total sales of the games')
plt.show()


# Feature Selection with correlation heatmap:
sns.heatmap(train_data.drop(['EU_Sales', 'Total_Sales'], 1).corr(), annot=True, cmap='coolwarm')
plt.show()


# Discretization of X_train['NA_Sales', 'JP_Sales', 'Other_Sales', 'EU_Sales']
for feature in ['NA_Sales', 'JP_Sales', 'Other_Sales']:
    X_train['cut_{}'.format(feature)] = pd.cut(X_train[feature], bins=5, labels=range(1, 6))
y_train['cut_EU_Sales'] = pd.cut(y_train['EU_Sales'], bins=5, labels=range(1, 6))

# Sum of all sales
X_train['sales_sum'] = X_train['NA_Sales'] + X_train['JP_Sales'] + X_train['Other_Sales']

# weighted avg of critics score and user score
X_train['avg_critic'] = (X_train['Critic_Score'] * X_train['Critic_Count'] + X_train['User_Score'] * X_train[
    'User_Count']) / (X_train['Critic_Count'] + X_train['User_Count'])

# share of sales for same year sales
for year in range(X_train['Year_of_Release'].min(), X_train['Year_of_Release'].max() + 1):
    year_sum = X_train[X_train['Year_of_Release'] == year][['NA_Sales', 'JP_Sales', 'Other_Sales']].sum().sum()
    X_train.loc[X_train['Year_of_Release'] == year, 'share_of_sales'] = X_train[['NA_Sales', 'JP_Sales',
                                                                                 'Other_Sales']].sum(
        axis=1) / year_sum

# Scale numerical columns with min_max
min_max_scaler = MinMaxScaler()
for feature in ['NA_Sales', 'JP_Sales', 'Other_Sales', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']:
    X_train[feature] = min_max_scaler.fit_transform(X_train[[feature]])

# Drop 'Name' column
X_train = X_train.drop('Name', axis=1)

# Assign all samples where 'Reviewed'=No to 'Rating'='Not Rated' and drop 'Reviewed' column
X_train.loc[X_train['Reviewed'] == 'NO', 'Rating'] = 'Not Rated'
X_train = X_train.drop('Reviewed', axis=1)

