import pandas as pd
import colorsys
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def washing_algorithm():
    # 소재 데이터 (소재: [care_label, washing temperature, dehydration_type])
    fabric_types = {'Cotton': ['Normal', 30, 'Light'],
                    'Polyester': ['Normal', 30, 'X'],
                    'Denim': ['Normal', 30, 'Mid'],
                    'Knit': ['Normal', 30, 'X'],
                    'Nylon': ['Normal', 30, 'Light'],
                    'Napping': ['Normal', 30, 'X'],
                    'Canvas': ['Normal', 30, 'Light'],
                    'Pique': ['Normal', 30, 'Light'],
                    'Rayon': ['Dry', 'X', 'X'],
                    'Silk': ['Dry', 'X', 'X'],
                    'Leather': ['Dry', 'X', 'X'],
                    'Pur': ['Dry', 'X', 'X'],
                    'Wool': ['Hand wash only', 'X', 'X'],
                    'Acrylic': ['Hand wash only', 'X', 'Light'],
                    'Linen': ['Hand wash only', 'X', 'X']}

    dataset = pd.read_csv('clothes_dataset.csv')

    data_num = dataset.shape[0]

    fabric_type = dataset['fabric_type']
    care_label = [fabric_types[fabric][0] for fabric in dataset['fabric_type']]
    water_temperature = [fabric_types[fabric][1] for fabric in dataset['fabric_type']]
    dehydration_type = [fabric_types[fabric][2] for fabric in dataset['fabric_type']]

    r = dataset['r']
    g = dataset['g']
    b = dataset['b']

    color_r = []
    color_g = []
    color_b = []

    h_list = []
    s_list = []
    v_list = []

    for i in range(data_num):
        color_r.append(r[i]/255)
        color_g.append(g[i]/255)
        color_b.append(b[i]/255)

        h, s, v = colorsys.rgb_to_hsv(color_r[i], color_g[i], color_b[i])

        h *= 360
        s *= 100
        v *= 100

        h_list.append(h)
        s_list.append(s)
        v_list.append(v)

    clothes_data = pd.DataFrame({
        'fabric_type': fabric_type,
        'care_label': care_label,
        'water_temperature': water_temperature,
        'dehydration_type': dehydration_type,
        'r': r,
        'g': g,
        'b': b,
        'h': h_list,
        's': s_list,
        'v': v_list
    })

    # Save the clothing dataset to a CSV file
    clothes_data.to_csv('clothes_dataset.csv', index=False)

    dataset = pd.read_csv('clothes_dataset.csv')
    dry_cleaning = dataset[dataset['care_label'].isin(['Dry'])]
    handwash_only = dataset[dataset['care_label'].isin(['Hand wash only'])]

    dry_cleaning_df = pd.DataFrame(dry_cleaning)
    handwash_only_df = pd.DataFrame(handwash_only)

    dry_cleaning_df.to_csv('dry_cleaning.csv', index=False)
    handwash_only_df.to_csv('handwash_only.csv', index=False)

    dataset = dataset[~dataset['care_label'].isin(['Dry', 'Hand wash only'])]

    data_num = dataset.shape[0]

    # Clustering
    hsv_data = dataset[['h', 's', 'v']]

    # Perform clustering with 1 cluster
    kmeans_1 = KMeans(n_clusters=1, random_state=42)
    cluster_labels_1 = kmeans_1.fit_predict(hsv_data)

    try:
        silhouette_score_1 = silhouette_score(hsv_data, cluster_labels_1)
    except ValueError:
        silhouette_score_1 = -1.0  # Set a low silhouette score for 1 cluster

    # Perform clustering with 2 clusters
    kmeans_2 = KMeans(n_clusters=2, random_state=42)
    cluster_labels_2 = kmeans_2.fit_predict(hsv_data)
    silhouette_score_2 = silhouette_score(hsv_data, cluster_labels_2)

    # Determine the optimal number of clusters
    if silhouette_score_1 > silhouette_score_2:
        num_clusters = 1
        cluster_labels = cluster_labels_1
    else:
        num_clusters = 2
        cluster_labels = cluster_labels_2

    # Add the cluster labels to the dataset
    dataset['cluster'] = cluster_labels

    basket_0_list = []
    basket_1_list = []

    # print("Cluster Results:")
    for index, row in dataset.iterrows():
        # print(f"Data point {index}: Cluster {row['cluster']} - Fabric Type: {row['fabric_type']}")
        if row['cluster'] == 0:
            basket_0_list.append(row)
        else:
            basket_1_list.append(row)

    basket_0_df = pd.DataFrame(basket_0_list)
    basket_1_df = pd.DataFrame(basket_1_list)

    basket_0_df.to_csv('basket_0.csv', index=False)
    basket_1_df.to_csv('basket_1.csv', index=False)

    result = []

    if num_clusters >= 2:
        basket_0 = pd.read_csv('basket_0.csv')
        basket_1 = pd.read_csv('basket_1.csv')

        basket_0_dehydration_1 = basket_0[basket_0['dehydration_type'].isin(['Light', 'Mid'])]
        basket_0_dehydration_0 = basket_0[basket_0['dehydration_type'].isin(['X'])]

        basket_1_dehydration_1 = basket_1[basket_1['dehydration_type'].isin(['Light', 'Mid'])]
        basket_1_dehydration_0 = basket_1[basket_1['dehydration_type'].isin(['X'])]

        water_temperature_basket_0_dehydration_1 = basket_0_dehydration_1['water_temperature'].astype(int)
        avg_water_temperature_basket_0_dehydration_1 = water_temperature_basket_0_dehydration_1.sum() // \
                                                       basket_0_dehydration_1.shape[0] if basket_0_dehydration_1.shape[
                                                                                              0] > 0 else 0
        indices_basket_0_dehydration_1 = basket_0_dehydration_1.index.tolist()

        water_temperature_basket_0_dehydration_0 = basket_0_dehydration_0['water_temperature'].astype(int)
        avg_water_temperature_basket_0_dehydration_0 = water_temperature_basket_0_dehydration_0.sum() // \
                                                       basket_0_dehydration_0.shape[0] if basket_0_dehydration_0.shape[
                                                                                              0] > 0 else 0
        indices_basket_0_dehydration_0 = basket_0_dehydration_0.index.tolist()

        water_temperature_basket_1_dehydration_1 = basket_1_dehydration_1['water_temperature'].astype(int)
        avg_water_temperature_basket_1_dehydration_1 = water_temperature_basket_1_dehydration_1.sum() // \
                                                       basket_1_dehydration_1.shape[0] if basket_1_dehydration_1.shape[
                                                                                              0] > 0 else 0
        indices_basket_1_dehydration_1 = basket_1_dehydration_1.index.tolist()

        water_temperature_basket_1_dehydration_0 = basket_1_dehydration_0['water_temperature'].astype(int)
        avg_water_temperature_basket_1_dehydration_0 = water_temperature_basket_1_dehydration_0.sum() // \
                                                       basket_1_dehydration_0.shape[0] if basket_1_dehydration_0.shape[
                                                                                              0] > 0 else 0
        indices_basket_1_dehydration_0 = basket_1_dehydration_0.index.tolist()

        result.append({
            'basket': 'basket_0',
            'dehydration': 'Light',
            'average_wash_temperature': int(avg_water_temperature_basket_0_dehydration_1),
            'indices': [basket_0.loc[indices_basket_0_dehydration_1, 'fabric_type'].tolist(), basket_0.loc[indices_basket_0_dehydration_1, 'r'].tolist(), basket_0.loc[indices_basket_0_dehydration_1, 'g'].tolist(), basket_0.loc[indices_basket_0_dehydration_1, 'b'].tolist()]
        })

        result.append({
            'basket': 'basket_0',
            'dehydration': 'X',
            'average_wash_temperature': int(avg_water_temperature_basket_0_dehydration_0),
            'indices': [basket_0.loc[indices_basket_0_dehydration_0, 'fabric_type'].tolist(), basket_0.loc[indices_basket_0_dehydration_0, 'r'].tolist(), basket_0.loc[indices_basket_0_dehydration_0, 'g'].tolist(), basket_0.loc[indices_basket_0_dehydration_0, 'b'].tolist()]
        })

        result.append({
            'basket': 'basket_1',
            'dehydration': 'Light',
            'average_wash_temperature': int(avg_water_temperature_basket_1_dehydration_1),
            'indices': [basket_1.loc[indices_basket_1_dehydration_1, 'fabric_type'].tolist(), basket_1.loc[indices_basket_1_dehydration_1, 'r'].tolist(), basket_1.loc[indices_basket_1_dehydration_1, 'g'].tolist(), basket_1.loc[indices_basket_1_dehydration_1, 'b'].tolist()]
        })

        result.append({
            'basket': 'basket_1',
            'dehydration': 'X',
            'average_wash_temperature': int(avg_water_temperature_basket_1_dehydration_0),
            'indices': [basket_1.loc[indices_basket_1_dehydration_0, 'fabric_type'].tolist(), basket_1.loc[indices_basket_1_dehydration_0, 'r'].tolist(), basket_1.loc[indices_basket_1_dehydration_0, 'g'].tolist(), basket_1.loc[indices_basket_1_dehydration_0, 'b'].tolist()]
        })

    else:
        print("옷을 분리 해서 세탁할 필요 없습니다.")

        dehydration_1 = dataset[dataset['dehydration_type'].isin(['Light', 'Mid'])]
        dehydration_0 = dataset[dataset['dehydration_type'].isin(['X'])]
        water_temperature = clothes_data['water_temperature'].astype(int)
        avg_water_temperature = water_temperature.sum() // data_num
        indices = dataset.index.tolist()

        if dehydration_1.shape[0] > 0:
            result.append({
                'basket': 'basket_0',
                'dehydration': 'Light',
                'average_wash_temperature': int(avg_water_temperature),
                'indices': [dataset.loc[indices, 'fabric_type'].tolist(), dataset.loc[indices, 'r'].tolist(), dataset.loc[indices, 'g'].tolist(), dataset.loc[indices, 'b'].tolist()]
            })
        elif dehydration_0.shape[0] > 0:
            result.append({
                'basket': 'basket_0',
                'dehydration': 'X',
                'average_wash_temperature': int(avg_water_temperature),
                'indices': [dataset.loc[indices, 'fabric_type'].tolist(), dataset.loc[indices, 'r'].tolist(), dataset.loc[indices, 'g'].tolist(), dataset.loc[indices, 'b'].tolist()]
            })

    if dry_cleaning.shape[0] > 0:
        dry_cleaning_csv = pd.read_csv('dry_cleaning.csv')
        indices = dry_cleaning_csv.index.tolist()
        result.append({
            'basket': 'dry_cleaning',
            'dehydration': 'X',
            'indices': [dry_cleaning_csv.loc[indices, 'fabric_type'].tolist(), dry_cleaning_csv.loc[indices, 'r'].tolist(), dry_cleaning_csv.loc[indices, 'g'].tolist(), dry_cleaning_csv.loc[indices, 'b'].tolist()]
        })
    if handwash_only.shape[0] > 0:
        handwash_only_csv = pd.read_csv('handwash_only.csv')
        indices = handwash_only_csv.index.tolist()
        result.append({
            'basket': 'handwash_only',
            'dehydration': 'X',
            'indices': [handwash_only_csv.loc[indices, 'fabric_type'].tolist(), handwash_only_csv.loc[indices, 'r'].tolist(), handwash_only_csv.loc[indices, 'g'].tolist(), handwash_only_csv.loc[indices, 'b'].tolist()]
        })

    print(result)
    return result
