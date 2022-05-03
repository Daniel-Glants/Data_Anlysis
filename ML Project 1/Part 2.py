import numbers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, confusion_matrix, silhouette_score, davies_bouldin_score
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, KBinsDiscretizer


def load_data():
    x_test = pd.read_csv('X_test.csv')
    x_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    y_test = pd.read_csv('y_test.csv')
    return x_train, x_test, y_train, y_test


def print_data(data):
    pd.set_option('display.width', 320)
    pd.set_option('display.max_columns', 320)
    print(data.head())


def discretize(data, n):
    kbins = KBinsDiscretizer(n_bins=n, encode='ordinal', strategy='quantile').fit(data)
    return pd.DataFrame(kbins.transform(data))[0]


def get_k_fold():
    return KFold(n_splits=10, shuffle=True, random_state=123)


def encode_dum(data, col_list):
    data_encoded = pd.get_dummies(data, columns=col_list)
    for col in data_encoded.columns[(len(data.columns)-len(col_list)-1)::]:
        data_encoded[col] = data_encoded[col].replace({0: -1})
    return data_encoded


def prep_data(x_train, x_test):
    x_train['sales_sum'] = x_train['NA_Sales'] + x_train['JP_Sales'] + x_train['Other_Sales']
    x_test['sales_sum'] = x_test['NA_Sales'] + x_test['JP_Sales'] + x_test['Other_Sales']

    for year in range(min([x_train['Year_of_Release'].min(), x_test['Year_of_Release'].min()]),
                      max([x_train['Year_of_Release'].max(), x_test['Year_of_Release'].max()]) + 1):
        year_sum_train = x_train[x_train['Year_of_Release'] == year][
            ['NA_Sales', 'JP_Sales', 'Other_Sales']].sum().sum()
        year_sum_test = x_test[x_test['Year_of_Release'] == year][['NA_Sales', 'JP_Sales', 'Other_Sales']].sum().sum()
        year_sum = year_sum_train + year_sum_test
        x_train.loc[x_train['Year_of_Release'] == year, 'share_of_sales'] = x_train[['NA_Sales', 'JP_Sales',
                                                                                     'Other_Sales']].sum(
            axis=1) / year_sum
        x_test.loc[x_test['Year_of_Release'] == year, 'share_of_sales'] = x_test[['NA_Sales', 'JP_Sales',
                                                                                  'Other_Sales']].sum(axis=1) / year_sum

    # Discretization of X_train['NA_Sales', 'JP_Sales', 'Other_Sales']
    for feature in ['NA_Sales', 'JP_Sales', 'Other_Sales', 'sales_sum', 'Year_of_Release', 'share_of_sales']:
        x_train[feature] = discretize(x_train[[feature]], 10) + 1
        x_test[feature] = discretize(x_test[[feature]], 10) + 1

    # weighted avg of critics score and user score
    x_train['avg_critic'] = (x_train['Critic_Score'] * x_train['Critic_Count'] + x_train['User_Score'] * x_train[
        'User_Count']) / (x_train['Critic_Count'] + x_train['User_Count'])
    x_test['avg_critic'] = (x_test['Critic_Score'] * x_test['Critic_Count'] + x_test['User_Score'] * x_test[
        'User_Count']) / (x_test['Critic_Count'] + x_test['User_Count'])

    # Assign all samples where 'Reviewed'=No to 'Rating'='Not Rated' and drop 'Reviewed' column
    x_train.loc[x_train['Reviewed'] == 'NO', 'Rating'] = 'Not Rated'
    x_train = x_train.drop('Reviewed', axis=1)
    x_test.loc[x_test['Reviewed'] == 'NO', 'Rating'] = 'Not Rated'
    x_test = x_test.drop('Reviewed', axis=1)

    x_train = x_train.drop(['Name', "Publisher", "Developer"], axis=1)
    x_test = x_test.drop(['Name', "Publisher", "Developer"], axis=1)

    x_train = encode_dum(x_train, ['Platform', 'Genre'])
    x_test = encode_dum(x_test, ['Platform', 'Genre'])

    encoder = LabelEncoder().fit(pd.concat([x_train['Rating'], x_test['Rating']]).unique())
    x_train['Rating'] = encoder.transform(x_train['Rating'])
    x_train['Rating'] = x_train['Rating'].replace(0, -1)
    x_test['Rating'] = encoder.transform(x_test['Rating'])
    x_test['Rating'] = x_test['Rating'].replace(0, -1)

    return x_train, x_test


def new_prep_data(x_train, x_test):
    x_train['sales_sum'] = x_train['NA_Sales'] + x_train['JP_Sales'] + x_train['Other_Sales']
    x_test['sales_sum'] = x_test['NA_Sales'] + x_test['JP_Sales'] + x_test['Other_Sales']

    for year in range(min([x_train['Year_of_Release'].min(), x_test['Year_of_Release'].min()]),
                      max([x_train['Year_of_Release'].max(), x_test['Year_of_Release'].max()]) + 1):
        year_sum_train = x_train[x_train['Year_of_Release'] == year][
            ['NA_Sales', 'JP_Sales', 'Other_Sales']].sum().sum()
        year_sum_test = x_test[x_test['Year_of_Release'] == year][['NA_Sales', 'JP_Sales', 'Other_Sales']].sum().sum()
        year_sum = year_sum_train + year_sum_test
        x_train.loc[x_train['Year_of_Release'] == year, 'share_of_sales'] = x_train[['NA_Sales', 'JP_Sales',
                                                                                     'Other_Sales']].sum(
            axis=1) / year_sum
        x_test.loc[x_test['Year_of_Release'] == year, 'share_of_sales'] = x_test[['NA_Sales', 'JP_Sales',
                                                                                  'Other_Sales']].sum(axis=1) / year_sum

    # Discretization of X_train['NA_Sales', 'JP_Sales', 'Other_Sales']
    for feature in ['NA_Sales', 'JP_Sales', 'Other_Sales', 'sales_sum', 'Year_of_Release', 'share_of_sales']:
        x_train[feature] = discretize(x_train[[feature]], 5) + 1
        x_test[feature] = discretize(x_test[[feature]], 5) + 1

    # weighted avg of critics score and user score
    x_train['avg_critic'] = (x_train['Critic_Score'] * x_train['Critic_Count'] + x_train['User_Score'] * x_train[
        'User_Count']) / (x_train['Critic_Count'] + x_train['User_Count'])
    x_test['avg_critic'] = (x_test['Critic_Score'] * x_test['Critic_Count'] + x_test['User_Score'] * x_test[
        'User_Count']) / (x_test['Critic_Count'] + x_test['User_Count'])

    # Assign all samples where 'Reviewed'=No to 'Rating'='Not Rated' and drop 'Reviewed' column
    x_train.loc[x_train['Reviewed'] == 'NO', 'Rating'] = 'Not Rated'
    x_train = x_train.drop('Reviewed', axis=1)
    x_test.loc[x_test['Reviewed'] == 'NO', 'Rating'] = 'Not Rated'
    x_test = x_test.drop('Reviewed', axis=1)

    x_train = x_train.drop(['Name', "Publisher", "Developer"], axis=1)
    x_test = x_test.drop(['Name', "Publisher", "Developer"], axis=1)

    encoder = LabelEncoder().fit(pd.concat([x_train['Rating'], x_test['Rating']]).unique())
    x_train['Rating'] = encoder.transform(x_train['Rating'])
    x_train['Rating'] = x_train['Rating'].replace(0, -1)
    x_test['Rating'] = encoder.transform(x_test['Rating'])
    x_test['Rating'] = x_test['Rating'].replace(0, -1)

    encoder = LabelEncoder().fit(pd.concat([x_train['Platform'], x_test['Platform']]).unique())
    x_train['Platform'] = encoder.transform(x_train['Platform'])
    x_train['Platform'] = x_train['Platform'].replace(0, -1)
    x_test['Platform'] = encoder.transform(x_test['Platform'])
    x_test['Platform'] = x_test['Platform'].replace(0, -1)

    encoder = LabelEncoder().fit(pd.concat([x_train['Genre'], x_test['Genre']]).unique())
    x_train['Genre'] = encoder.transform(x_train['Genre'])
    x_train['Genre'] = x_train['Genre'].replace(0, -1)
    x_test['Genre'] = encoder.transform(x_test['Genre'])
    x_test['Genre'] = x_test['Genre'].replace(0, -1)


    return x_train, x_test


def basic_ann_model(x_train, y_train):
    cv = get_k_fold()
    np.random.seed(123)
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    model = MLPClassifier(random_state=1)
    scores=[cross_val_score(model,x_train,y_train,cv=cv)]
    print("The Avvarege Score of the CV training is: " ,scores[0].mean())
    return model


def plot_grid_search_validation_curve(grid, param_to_vary, title='Validation Curve', ylim=None, xlim=None, log=None):

    df_cv_results = pd.DataFrame(grid.cv_results_)
    train_scores_mean = df_cv_results['mean_train_score']
    valid_scores_mean = df_cv_results['mean_test_score']
    train_scores_std = df_cv_results['std_train_score']
    valid_scores_std = df_cv_results['std_test_score']

    param_cols = [c for c in df_cv_results.columns if c[:6] == 'param_']
    param_ranges = [grid.param_grid[p[6:]] for p in param_cols]
    param_ranges_lengths = [len(pr) for pr in param_ranges]

    train_scores_mean = np.array(train_scores_mean).reshape(*param_ranges_lengths)
    valid_scores_mean = np.array(valid_scores_mean).reshape(*param_ranges_lengths)
    train_scores_std = np.array(train_scores_std).reshape(*param_ranges_lengths)
    valid_scores_std = np.array(valid_scores_std).reshape(*param_ranges_lengths)

    param_to_vary_idx = param_cols.index('param_{}'.format(param_to_vary))

    slices = []
    for idx, param in enumerate(grid.best_params_):
        if (idx == param_to_vary_idx):
            slices.append(slice(None))
            continue
        best_param_val = grid.best_params_[param]
        idx_of_best_param = 0
        if isinstance(param_ranges[idx], np.ndarray):
            idx_of_best_param = param_ranges[idx].tolist().index(best_param_val)
        else:
            idx_of_best_param = param_ranges[idx].index(best_param_val)
        slices.append(idx_of_best_param)

    train_scores_mean = train_scores_mean[tuple(slices)]
    valid_scores_mean = valid_scores_mean[tuple(slices)]
    train_scores_std = train_scores_std[tuple(slices)]
    valid_scores_std = valid_scores_std[tuple(slices)]

    plt.clf()

    plt.title(title)
    plt.xlabel(param_to_vary)
    plt.ylabel('Score')

    if (ylim is None):
        plt.ylim(0.0, 1.1)
    else:
        plt.ylim(*ylim)

    if (not (xlim is None)):
        plt.xlim(*xlim)

    lw = 2

    plot_fn = plt.plot
    if log:
        plot_fn = plt.semilogx

    param_range = param_ranges[param_to_vary_idx]
    if (not isinstance(param_range[0], numbers.Number)):
        param_range = [str(x) for x in param_range]
    plot_fn(param_range, train_scores_mean, label='Training score', color='r',
            lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color='r', lw=lw)
    plot_fn(param_range, valid_scores_mean, label='Cross-validation score',
            color='b', lw=lw)
    plt.fill_between(param_range, valid_scores_mean - valid_scores_std,
                     valid_scores_mean + valid_scores_std, alpha=0.1,
                     color='b', lw=lw)

    plt.legend(loc='lower right')

    plt.show()


def evaluate(model, x_train, y_train):
    scaler = StandardScaler()
    scaler.fit(x_train)
    model.fit(scaler.transform(x_train),y_train)

    #   Accuracy & confusion Matrix
    print(f"The Model's Accuracy on the Test Data: {accuracy_score(y_true=y_train, y_pred=model.predict(scaler.transform(x_train))):.3f}")
    print( confusion_matrix(y_true=y_train, y_pred=model.predict(scaler.transform(x_train))))

    #   Loss Curve
    plt.plot(model.loss_curve_)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()


def hype_p_tuning(estimator, x_train, y_train):
    cv = get_k_fold()
    np.random.seed(123)
    scaler = StandardScaler()
    scaler.fit(x_train)

    hidden_layer_sizesARRAY = np.arange(5,20,2)

    param_search = {'hidden_layer_sizes': hidden_layer_sizesARRAY,
                    'activation': ['identity', 'logistic', 'tanh', 'relu'],
                    'solver': ['lbfgs', 'sgd', 'adam'],
                    'learning_rate': ['constant', 'invscaling', 'adaptive'],
                    'alpha': np.arange(0.02, 0.1, 0.02)
                    }

    grid_search = GridSearchCV(n_jobs = -1,
                               estimator= estimator,
                               param_grid=param_search,
                               refit=True,
                               cv=cv,
                               return_train_score=True)
    grid_search.fit(scaler.transform(x_train), y_train)
    best_model = grid_search.best_estimator_
    acc = grid_search.best_score_
    bestParams = grid_search.best_params_
    return acc, best_model, bestParams, grid_search


def print2Smallest(arr):
    largest = arr[0]
    lowest = arr[0]
    largest2 = None
    lowest2 = None
    for item in arr[1:]:
        if item > largest:
            largest2 = largest
            largest = item
        elif largest2 == None or largest2 < item:
            largest2 = item
        if item < lowest:
            lowest2 = lowest
            lowest = item
        elif lowest2 == None or lowest2 > item:
            lowest2 = item
    return lowest , lowest2


def least_confident_calss(x_train,best_model):
    scaler = StandardScaler()
    scaler.fit(x_train)
    res = best_model.predict_proba(scaler.transform(x_train))
    class_0_first = res[:,0].tolist().index(print2Smallest(res[:,0][(res[:,0]-0.5)>0])[0])
    class_0_second = res[:,0].tolist().index(print2Smallest(res[:,0][(res[:,0]-0.5)>0])[1])
    class_1_first = res[:,1].tolist().index(print2Smallest(res[:,1][(res[:,1]-0.5)>0])[0])
    class_1_second = res[:,1].tolist().index(print2Smallest(res[:,1][(res[:,1]-0.5)>0])[1])
    print("The first least confident prediction is for class '0' is the " + str(class_0_first) +
          "'th Sample with probability of "+str(print2Smallest(res[:,0][(res[:,0]-0.5)>0])[0]))
    print("The second least confident prediction is for class '0' is the " + str(class_0_second)+
          "'th Sample with probability of "+str(print2Smallest(res[:,0][(res[:,0]-0.5)>0])[1]))
    print("The first least confident prediction is for class '1' is the " + str(class_1_first) +
          "'th Sample with probability of "+str(print2Smallest(res[:,1][(res[:,1]-0.5)>0])[0]))
    print("The second least confident prediction is for class '1' is the " + str(class_1_second) +
          "'th Sample with probability of "+str(print2Smallest(res[:,1][(res[:,1]-0.5)>0])[1]))


def pca_on_data(x_train,y_train):
    scaler = StandardScaler()
    scaler.fit(x_train)
    pca = PCA(n_components=2)
    pca.fit(scaler.transform(x_train))
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.sum())
    data_pca = pca.transform(scaler.transform(x_train))
    data_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
    data_pca['y'] = y_train
    plt.title("Origin Data after PCA")
    sns.scatterplot(x='PC1', y='PC2', hue='y', data=data_pca, palette="muted")
    plt.show()


def basic_k_means_training(x_train):
    scaler = StandardScaler()
    scaler.fit(x_train)
    kmeans = KMeans(n_clusters=2, random_state=123)
    kmeans.fit(scaler.transform(x_train))
    data_pca['cluster'] = kmeans.predict(scaler.transform(x_train))
    sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=data_pca, palette="muted")
    plt.title("Kmeans - K=2")
    plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1], marker='+',
                s=100, color='magenta')
    plt.show()


def choose_beset_k(x_train):
    scaler = StandardScaler()
    scaler.fit(x_train)
    iner_list = []
    dbi_list = []
    sil_list = []

    for n_clusters in np.arange(2, 10, 1):
        kmeans = KMeans(n_clusters=n_clusters, max_iter=300, n_init=10, random_state=123)
        kmeans.fit(scaler.transform(x_train))
        assignment = kmeans.predict(scaler.transform(x_train))

        iner = kmeans.inertia_
        sil = silhouette_score(scaler.transform(x_train), assignment)
        dbi = davies_bouldin_score(scaler.transform(x_train), assignment)

        dbi_list.append(dbi)
        sil_list.append(sil)
        iner_list.append(iner)

    plt.figure()
    plt.plot(range(2, 10, 1), iner_list, marker='o')
    plt.title("Inertia")
    plt.xlabel("Number of clusters")
    plt.show()

    plt.figure()
    plt.plot(range(2, 10, 1), sil_list, marker='o')
    plt.title("Silhouette")
    plt.xlabel("Number of clusters")
    plt.show()

    plt.figure()
    plt.plot(range(2, 10, 1), dbi_list, marker='o')
    plt.title("Davies-bouldin")
    plt.xlabel("Number of clusters")
    plt.show()


def get_second(elem):
    return elem[1]


def main():
    x_train, x_test, y_train, y_test = load_data()
    y_train = discretize(y_train, 2)
    x_train, x_test = prep_data(x_train,x_test)
    pca_on_data(x_train, y_train)
    # ----------------------
    #          ANN
    ----------------------
    ann_model = basic_ann_model(x_train, y_train)
    evaluate(ann_model, x_train, y_train)

    acc, best_model, bestParams, grid_search = hype_p_tuning(ann_model, x_train, y_train)

    #Print Grid search parameters graphs
    plot_grid_search_validation_curve(grid_search, 'activation')
    plot_grid_search_validation_curve(grid_search, 'alpha')
    plot_grid_search_validation_curve(grid_search, 'hidden_layer_sizes')
    plot_grid_search_validation_curve(grid_search, 'learning_rate')
    plot_grid_search_validation_curve(grid_search, 'solver')

    # ----------------------
    #          K-Means
    # ----------------------


    basic_k_means_training(x_train,y_train)
    choose_beset_k(x_train)

    y_train_new_L =discretize(pd.read_csv('y_train.csv'),10)
    evaluate(best_model,x_train,y_train_new_L)

    # ----------------------------------------
    #          Selected model Improvements
    # ----------------------------------------

    new_x_train, new_x_test = new_prep_data(pd.read_csv('X_train.csv'),pd.read_csv('X_test.csv'))
