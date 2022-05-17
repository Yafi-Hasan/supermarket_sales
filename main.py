# Supermarket-Sales Dataset
# Member-Type Customer Optimization
# How to optimize Member-Type Customer total transaction
# How to build a model that can predict potential member-type customer, based on supermarket sales data

import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
pd.set_option('display.precision',3)


def data_summary(dataframe):
    summary = pd.DataFrame()
    summary['Feature Names'] = dataframe.columns.values
    summary['Data Types'] = dataframe.dtypes.values
    summary['Rows'] = len(dataframe)
    summary['Duplicate Rows'] = dataframe.duplicated().sum()
    summary['Num Missing Val'] = dataframe.isnull().sum().values
    summary['% Missing Val'] = summary['Num Missing Val'] / summary['Rows']
    summary['Num Unique Val'] = dataframe.nunique().values
    summary['% Unique Val'] = summary['Num Unique Val'] / summary['Rows']
    return summary


def box_plot(dataframe, x_values, y_values, rotation=0, hue=None, figsize=(10,6), title=None, savefig=False):
    sns.set(rc={'figure.dpi': 100, 'savefig.dpi': 100, 'figure.figsize': figsize})
    a = sns.boxplot(x=x_values, y=y_values, hue=hue, data=dataframe)
    a.set_xticklabels(a.get_xticklabels(), rotation=rotation)
    if savefig and title is not None:
        a.set_title(title)
        plt.savefig(title + '.png')
    plt.close()


def bar_plot(dataframe, x_values, y_values, rotation=0, hue=None, title=None, figsize=(10,6), savefig=False):
    sns.set(rc={'figure.dpi': 100, 'savefig.dpi': 100, 'figure.figsize': figsize})
    a = sns.barplot(data=dataframe, x=x_values, y=y_values, hue=hue)
    a.set_xticklabels(a.get_xticklabels(),rotation=rotation)
    if savefig and title is not None:
        a.set_title(title)
        plt.savefig(title + '.png')
    plt.close()


def sbs_barplot(dataframe, x_values, hue, rename, rotation=0, title=None, figsize=(10,6), savefig=False):
    dataset = dataframe.groupby([hue])[x_values].value_counts(normalize=True).rename(rename).reset_index()
    sns.set(rc={'figure.dpi': 100, 'savefig.dpi': 100, 'figure.figsize': figsize})
    a = sns.barplot(x=x_values, y=rename, hue=hue, data=dataset)
    a.set_xticklabels(a.get_xticklabels(),rotation=rotation)
    if savefig and title is not None:
        a.set_title(title)
        plt.savefig(title + '.png')
    plt.close()


def stacked_plot(dataframe, feature, kind='barh', title=None, savefig=False):
    hold = dataframe[feature].value_counts(normalize=True)
    dict = {}
    for i in range(len(hold.index)):
        dict[str(hold.index[i])] = hold[i]
    data = pd.DataFrame(data=dict, index=[feature])
    data.plot(kind=kind, stacked=True)
    plt.xlabel('% Transaction')
    if savefig and title is not None:
        plt.title(title)
        plt.savefig(title + '.png')
    plt.close()


def find_distance(array, cluster):
    distances = []
    for i in range(len(array)):
        dist = np.linalg.norm(array[i] - cluster)
        distances.append(dist)
    return distances


def feature_iv(dataframe_x, dataframe_y, max_bins):
    r = 0
    n_bins = max_bins
    while np.abs(r) < 1:
      try:
          df = pd.DataFrame({"X": dataframe_x, "Y": dataframe_y, "Bins": pd.qcut(dataframe_x, n_bins)})
          df_2 = df.groupby('Bins', as_index=True)
          r, p = spearmanr(df_2.mean().X, df_2.mean().Y)
          n_bins = n_bins - 1
      except Exception:
          n_bins = n_bins - 1

    if len(df_2) == 1:
        n = max_bins
        bins = []
        q_array = np.linspace(0,1,n)
        for q in q_array:
            bins.append(dataframe_x.quantile(q))

        bins = np.array(bins)

        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1] / 2

        df = pd.DataFrame({"X": dataframe_x, "Y": dataframe_y, "Bins": pd.cut(dataframe_x, np.unique(bins), include_lowest=True)})
        df_2 = df.groupby('Bins', as_index=True)

    df_3 = pd.DataFrame()
    df_3['Min_Value'] = df_2.min().X
    df_3['Max_Value'] = df_2.max().X
    df_3['Count'] = df_2.count().Y
    df_3['Event'] = df_2.sum().Y
    df_3['NonEvent'] = df_2.count().Y - df_2.sum().Y
    df_3 = df_3.reset_index(drop=True)
    df_3['Event_Rate'] = df_3.Event / df_3.Count
    df_3['NonEvent_Rate'] = df_3.NonEvent / df_3.Count
    df_3['IV'] = (df_3.Event_Rate - df_3.NonEvent_Rate) * np.log(df_3.Event_Rate / df_3.NonEvent_Rate)
    df_3 = df_3.replace([np.inf, -np.inf], 0)
    df_3.IV = df_3['IV'].sum()
    return df_3


def information_value(dataframe_x, dataframe_y, max_bins):
    iv = []
    for feature in dataframe_x.columns:
        iv.append(feature_iv(dataframe_x[feature], dataframe_y, max_bins).IV[0])

    df = pd.DataFrame({"Var_Name": dataframe_x.columns, "IV": iv})
    return df


def main():
    # Accessing file
    work_dir = os.getcwd()
    data_dir = os.path.join(work_dir, 'Dataset')
    filename = 'supermarket_sales.csv'
    df = pd.read_csv(os.path.join(data_dir,filename))
    summary = data_summary(df)
    summary.to_csv('Raw Summary.csv')

    # Formatting dataframe
    print('[INFO] Formatting dataframe...')
    df['City Branch'] = df['City'] + ' ' + df['Branch']
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df['Month'] = pd.to_datetime(df['Datetime']).dt.strftime('%B')
    df['Hour'] = pd.to_datetime(df['Datetime']).dt.hour
    df = df.drop(['City', 'Branch', 'Date', 'Time', 'gross margin percentage'], axis=1)
    new_sum = data_summary(df)
    new_sum.to_csv('Summary.csv')
    print(f'[INFO] Data Summary: \n{new_sum}')
    df_stats = df.describe()
    df_stats.to_csv('Stats.csv')
    print(f'[INFO] Data Stats: \n{df_stats}')

    # Missing value treatment
    # None, no missing value

    # Data distribution review
    df_cat = df[['Invoice ID', 'Customer type', 'City Branch', 'Gender', 'Product line', 'Payment', 'Month']]
    df_num = df[['Unit price', 'Quantity', 'Total', 'gross income', 'Hour', 'cogs', 'Tax 5%', 'Rating']]
    for feature in df_num.columns.values:
        box_plot(df, 'Customer type', feature, title=(feature + ' Distribution per Customer Type'), savefig=True)
    stacked_plot(df, 'Customer type', title='Customer Type Based Transaction Distribution', savefig=True)
    stacked_plot(df, 'City Branch', title='City Branch Based Transaction Distribution', savefig=True)
    stacked_plot(df, 'Gender', title='Gender Based Transaction Distribution', savefig=True)
    stacked_plot(df, 'Product line', title='Product Line Based Transaction Distribution', savefig=True)
    stacked_plot(df, 'Payment', title='Payment Method Based Transaction Distribution', savefig=True)
    stacked_plot(df, 'Month', title='Month Based Transaction Distribution', savefig=True)

    # Generating insight for building hypothesis
    ## Customer type analysis based on categorical feature
    print(f'[INFO] Generating insight...')
    sbs_barplot(df, 'Product line', 'Customer type', '% Transaction', rotation=15, title='Percentage Transaction per Customer Type Based on Product Line', figsize=(10,8.5), savefig=True)
    sbs_barplot(df, 'City Branch', 'Customer type', '% Transaction', title='Percentage Transaction per Customer Type Based on City Branch', savefig=True)
    sbs_barplot(df, 'Gender', 'Customer type', '% Transaction', title='Percentage Transaction per Customer Type Based on Gender', savefig=True)
    sbs_barplot(df, 'Payment', 'Customer type', '% Transaction', title='Percentage Transaction per Customer Type Based on Payment Method', savefig=True)
    sbs_barplot(df, 'Month', 'Customer type', '% Transaction', title='Percentage Transaction per Customer Type Based on Month', savefig=True)
    sbs_barplot(df, 'Hour', 'Customer type', '% Transaction', title='Percentage Transaction per Customer Type Based on Hour', savefig=True)
    sbs_barplot(df, 'Quantity', 'Customer type', '% Transaction', title='Percentage Transaction per Customer Type Based on Quantity', savefig=True)

    ## Binning Unit Price
    bins = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    bin_names = ['10-20 USD', '21-30 USD', '31-40 USD', '41-50 USD', '51-60 USD', '61-70 USD', '71-80 USD', '81-90 USD', '91-100 USD']
    df['Range Unit Price'] = pd.cut(df['Unit price'], bins=bins, labels=bin_names).astype(str)
    sbs_barplot(df, 'Range Unit Price', 'Customer type', '% Transaction',rotation=15, figsize=(10,7), title='Percentage Transaction per Customer Type Based on Quantity', savefig=True)

    # Outlier treatment
    print(f'[INFO] Treating outlier...')
    up_limit = df_num.quantile(0.95)
    low_limit = df_num.quantile(0.05)
    more_than = df_num > up_limit
    lower_than = df_num < low_limit
    df_num_mask = df_num.mask(more_than, up_limit, axis=1)
    df_num_mask = df_num_mask.mask(lower_than, low_limit, axis=1)
    df_out = pd.concat([df_cat, df_num_mask], axis=1)

    # Encoding categorical features
    print(f'[INFO] Encoding categorical features...')
    df_out_cat = df_out['Invoice ID']
    df_out_num = df_out.drop(['Invoice ID'], axis=1)
    df_out_num = pd.get_dummies(df_out_num)
    df_out = pd.concat([df_out_num, df_out_cat], axis=1)

    # Clustering member-type customer
    print(f'[INFO] Clustering...')
    ## Creating Member-Type Customer Dataframe
    df_member = df_out[df_out['Customer type_Member'] == 1]
    df_member_id = df_member['Invoice ID']
    df_member = df_member.drop(['Customer type_Member', 'Customer type_Normal', 'Invoice ID'], axis=1)
    df_notmember = df_out[df_out['Customer type_Normal'] == 1]
    df_notmember_id = df_notmember['Invoice ID']
    df_notmember = df_notmember.drop(['Customer type_Member', 'Customer type_Normal', 'Invoice ID'], axis=1)

    ## Standard Scaling
    scaler = StandardScaler()
    scaler.fit(df_member)

    ## Clustering
    inertias = []
    K = range(1,10)
    for k in K:
        kmean = KMeans(n_clusters=k, n_init=15, max_iter=400)
        kmean.fit(df_member)
        inertias.append(kmean.inertia_)
    plt.figure(figsize=(10,6))
    plt.plot(K, inertias, 'bo-')
    plt.title('Elbow Method Based on Inertia')
    plt.xlabel('Number of k')
    plt.ylabel('Inertia')
    plt.savefig('Elbow Method Based on Inertia.png')
    plt.close()
    kmean_model = KMeans(n_clusters=2, n_init=15, max_iter=400)
    pred = kmean_model.fit(df_member)
    cluster_1 = pred.cluster_centers_[0]
    cluster_2 = pred.cluster_centers_[1]
    labels = list(pred.labels_)
    unique, count = np.unique(labels, return_counts=True)
    cluster = pd.DataFrame(list(zip(unique, count)), columns=['Cluster', 'Number of Transaction'])
    cluster['Cluster'] = cluster['Cluster'] + 1

    ### Visualizing cluster
    plt.figure(figsize=(10,6))
    plt.bar(cluster['Cluster'], cluster['Number of Transaction'])
    plt.xlabel('Cluster')
    plt.xticks(np.arange(1,3))
    plt.ylabel('Number of Transaction')
    plt.title('Number of Member-Type Customer Transaction per Cluster')
    plt.savefig('Number of Member-Type Customer Transaction per Cluster.png')
    plt.close()

    ## Measuring distance to cluster
    dist_cluster_1 = find_distance(df_notmember.values, cluster_1)
    dist_cluster_2 = find_distance(df_notmember.values, cluster_2)

    ## Creating cluster distance dataframe
    ident = list(df_notmember_id)
    dist_clust = pd.DataFrame({'Invoice ID': ident,
                               'dist_cluster_1': dist_cluster_1,
                               'dist_cluster_2': dist_cluster_2})
    print(f'[INFO] Cluster information:\n{dist_clust.describe()}')
    not_potential_transaction_1 = dist_clust.loc[dist_clust['dist_cluster_1'] >= 465]
    not_potential_transaction_2 = dist_clust.loc[dist_clust['dist_cluster_2'] >= 465] #337
    not_potential_transaction = pd.concat([not_potential_transaction_1, not_potential_transaction_2], ignore_index=True)
    not_potential_transaction.drop_duplicates(subset='Invoice ID', keep='first', inplace=True)
    df_not_potential = pd.merge(df_out, not_potential_transaction, on=['Invoice ID'])
    df_potential = df_out[df_out['Customer type_Member'] == 1]
    data_frame = df_not_potential.append(df_potential)
    id = pd.Index(range(len(data_frame)))
    data_frame['Num ID'] = list(id)
    data_frame = data_frame.drop(['Invoice ID'], axis=1)
    data_frame = data_frame.sample(frac=1)

    # Weight of Evidence and Information Value
    print('[INFO] Calculating WoE and IV...')
    data_x = data_frame.drop(['Customer type_Member', 'Customer type_Normal'], axis=1)
    data_y = data_frame['Customer type_Member']
    df_iv = information_value(data_x, data_y, 20)
    df_iv.to_csv('Information Value Features.csv')
    df_iv = df_iv.loc[(df_iv['IV'] > 0.02) & (df_iv['IV'] < 0.5)]
    df_iv = df_iv.sort_values(['IV'])
    df_iv = df_iv.reset_index(drop=True)
    df_iv.to_csv('Selected IV Feature.csv')
    print('[INFO] Process completed.')


if __name__ == '__main__':
    main()
