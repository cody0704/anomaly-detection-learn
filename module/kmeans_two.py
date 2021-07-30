import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def kmeans(data):
    dx = pd.DataFrame(data['data'], columns=data['feature_names'])
    dy = pd.DataFrame(data['target'], columns=['MEDV'])
    df = pd.concat((dy, dx), axis=1)
    K_max = 20
    # calc coef
    scores = []
    for i in range(2, K_max + 1):
        scores.append(
            silhouette_score(
                df, KMeans(n_clusters=i).fit_predict(df)))

    # choice k value
    selected_K = scores.index(max(scores)) + 2
    print('K =', selected_K, '\n')

    # grouping
    kmeans = KMeans(n_clusters=selected_K)
    labels = kmeans.fit_predict(df)

    # import orignal data to grouping data
    lb = pd.DataFrame(labels, columns=['labels'])
    df = pd.concat((lb, df), axis=1)

    # print(df[['labels', 'MEDV']], '\n')
    # print('原始資料\n', df['MEDV'].describe(), '\n')

    df_group = []
    for i in range(selected_K):
        df_new = df[df['labels']==i]['MEDV']
        print(f'分類 {i + 1}\n', df_new.describe(), '\n')
        df_group.append(df_new)
    
    return df_group