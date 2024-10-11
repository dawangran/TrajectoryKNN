# -*- coding: utf-8 -*-
"""
@File    :   trajectoryknn.py
@Time    :   2024/08/01 
@Author  :   Dawn
@Version :   1.0
@Desc    :   Trajectoryknn for single cell
"""



from sklearn.neighbors import NearestNeighbors
from pyecharts import options as opts
from pyecharts.charts import Sankey
import pandas as pd
import numpy as np
import os



def most_frequent_with_positions(row):
    """
    Return a dictionary mapping the most frequent elements in the input Series to their positions.

    Args:
        row (pd.Series): Input pandas Series.

    Returns:
        dict: Dictionary mapping most frequent elements to their positions.
    """
    try:
        # Validate input
        if not isinstance(row, pd.Series):
            raise ValueError("Input must be a pandas Series.")
        
        counts = row.value_counts()
        max_count = counts.max()
        most_frequent_elements = counts[counts == max_count].index.tolist()
        
        positions_dict = {element: [] for element in most_frequent_elements}
        for element in most_frequent_elements:
            positions_dict[element] = row[row == element].index.tolist()
        
        return positions_dict
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}



def generate_labels(df_indices_replaced, df_distances):
    """
    Generate labels based on replaced indices and distances.

    Args:
        df_indices_replaced (pd.DataFrame): DataFrame with replaced indices.
        df_distances (pd.DataFrame): DataFrame with distances.

    Returns:
        list: List of generated labels.
    """
    try:
        label_list = []
        for index, row in df_indices_replaced.iterrows():
            if len(row["most_frequent"]) == 1:
                label_list.append(list(row["most_frequent"].keys())[0])
            else:
                dis_dict = {}
                for k, v in row["most_frequent"].items():
                    dis_list = df_distances.loc[index].to_list()
                    # dis = sum([dis_list[i] * (i+1) for i in v])
                    dis = sum([dis_list[i] for i in v])
                    dis_dict[k] = dis
                min_dis = min(dis_dict, key=dis_dict.get)
                label_list.append(min_dis)
        return label_list
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return []



def generate_distance(data_indices, data_distances):
    """
    Generate distances based on data indices and distances.

    Args:
        data_indices (pd.DataFrame): DataFrame with data indices.
        data_distances (pd.DataFrame): DataFrame with distances.

    Returns:
        list: List of generated distances.
    """
    try:
        dis_target_list = []
        for index, row in data_indices.iterrows():
            dis_list = data_distances.loc[index].to_list()
            dis_most_list = row["most_frequent"][row['target']]
            dis = np.mean([dis_list[i] for i in dis_most_list])
            dis_target_list.append(dis)
        return dis_target_list
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return []



def traject_knn(pca_data,meta_data,stage_list,label,cell_number=300,min_wight=5,knn_number=10,nmads=5):
    """
    Perform trajectory analysis using K-nearest neighbors (KNN) algorithm.

    Args:
    pca_data (pd.DataFrame): DataFrame containing PCA data.
    meta_data (pd.DataFrame): DataFrame containing metadata.
    stage_list (list): List of stages for trajectory analysis.
    label (str): Column name for labels.
    cell_number (int): Number of cells for sampling.
    min_wight (int): Minimum weight for filtering.
    knn_number (int): Number of nearest neighbors for KNN.
    nmads (int): Number of median absolute deviations for outlier filtering.

    Returns:
    data_cell (pd.DataFrame): Processed data for cells.
    data_cluster (pd.DataFrame): Processed data for clusters.
    data_indices (pd.DataFrame): Processed data for indices.
    data_distances (pd.DataFrame): Processed data for distances.
    data_barcode (pd.DataFrame): Processed data for barcodes.
    """
    
    # Initialize lists to store processed data
    data_list=[]
    df_indices_list=[]
    df_barcode_list=[]
    df_distances_list=[]
    
    # Iterate over stages for trajectory analysis
    for i in range(len(stage_list)-1):
        
        ref_meta=meta_data[meta_data['state']==stage_list[i]]
        target_meta=meta_data[meta_data['state']==stage_list[i+1]]
    
        # Extract metadata for reference and target stages
        grouped = target_meta.groupby(label)
        label_counts = dict(grouped.size())
        target_meta_list=[]
        for k,v  in label_counts.items():
            target_meta_sub=target_meta[target_meta[label]==k]
            if v <cell_number:
                target_meta_list.append(target_meta_sub)
            else:
                target_meta_sub=target_meta_sub.sample(n=cell_number, random_state=1)
                target_meta_list.append(target_meta_sub)
        target_meta=pd.concat(target_meta_list)
    
        ref_cell=ref_meta.index.to_list()
        target_cell=target_meta.index.to_list()
        ref_pca_data=pca_data.loc[ref_cell]
        target_pca_data=pca_data.loc[target_cell]
    
        # KNN model fitting
        nbrs = NearestNeighbors(n_neighbors=knn_number, algorithm='auto',radius=1).fit(target_pca_data)
        distances, indices = nbrs.kneighbors(ref_pca_data)
    
        # Calculate cell trajectories
        df_distances=pd.DataFrame(distances)
        seurat_clusters_dict={i:v for i,v in enumerate(target_meta[label])}
        barcode_dict={i:v for i,v in enumerate(target_meta['barcode'])}
        df_indices=pd.DataFrame(indices)
        df_indices_replaced = df_indices.replace(seurat_clusters_dict)
        df_barcode = df_indices.replace(barcode_dict)
        df_indices_replaced['most_frequent'] = df_indices_replaced.apply(most_frequent_with_positions, axis=1)
    
        target_list=generate_labels(df_indices_replaced,df_distances)
        df_indices_replaced['target']=target_list
        df_indices_replaced['barcode']=ref_cell
        df_indices_replaced['most_frequent_number']=[len(j[i]) for i,j in zip(df_indices_replaced['target'],df_indices_replaced['most_frequent'])]
        df_indices_replaced['stage']="{}_{}".format(stage_list[i],stage_list[i+1])
        
        sub_data=pd.DataFrame({"barcode":ref_cell,"source":ref_meta[label].to_list(),"target":target_list})
        sub_data['source_stage']=stage_list[i]
        sub_data['target_stage']=stage_list[i+1]
        
        df_barcode['barcode']=ref_cell
        df_barcode['stage']="{}_{}".format(stage_list[i],stage_list[i+1])
        df_distances['stage']="{}_{}".format(stage_list[i],stage_list[i+1])
        df_distances['barcode']=ref_cell
    
        # Filter out cells with low weights
        out_cell_1=list(df_indices_replaced[df_indices_replaced['most_frequent_number']<= min_wight].index)
        print("{}_{} : step1. filter weight cell {}".format(stage_list[i],stage_list[i+1],len(out_cell_1)))
        df_barcode=df_barcode[~df_barcode.index.isin(out_cell_1)]
        sub_data=sub_data[~sub_data.index.isin(out_cell_1)]
        df_indices_replaced=df_indices_replaced[~df_indices_replaced.index.isin(out_cell_1)]
        df_distances=df_distances[~df_distances.index.isin(out_cell_1)]

        # Filter out cells with large deviations
        df_indices_replaced['distance']= generate_distance(df_indices_replaced, df_distances)
        df_indices_replaced['abs_deviation'] = np.abs(df_indices_replaced['distance'] - np.median(df_indices_replaced['distance']))
        mads = nmads * np.median(df_indices_replaced['abs_deviation'])
        out_cell_2=list(df_indices_replaced[df_indices_replaced['abs_deviation']>mads].index)
        print("{}_{} : step2. filter distance cell {}".format(stage_list[i],stage_list[i+1],len(out_cell_2)))
        df_barcode=df_barcode[~df_barcode.index.isin(out_cell_2)]
        sub_data=sub_data[~sub_data.index.isin(out_cell_2)]
        df_indices_replaced=df_indices_replaced[~df_indices_replaced.index.isin(out_cell_2)]
        df_distances=df_distances[~df_distances.index.isin(out_cell_2)]
        
        # Append processed data to lists
        data_list.append(sub_data)
        df_indices_list.append(df_indices_replaced)
        df_barcode_list.append(df_barcode)
        df_distances_list.append(df_distances)
        
    # Combine results
    data_cell=pd.concat(data_list)
    data_cell['source_label']=["{}_{}".format(j,i) for i,j in zip(data_cell['source'],data_cell['source_stage'])]
    data_cell['target_label']=["{}_{}".format(j,i) for i,j in zip(data_cell['target'],data_cell['target_stage'])]
    data_cluster=data_cell.groupby(['source_label','target_label']).agg("count")[['barcode']]
    data_cluster=data_cluster.reset_index()
    data_indices=pd.concat(df_indices_list)
    data_distances=pd.concat(df_distances_list)
    data_barcode=pd.concat(df_barcode_list)

    return data_cell,data_cluster,data_indices,data_distances,df_barcode
        
        


def generate_sankey_data(data_cluster):
    """
    Generate a Sankey chart based on the data_cluster.

    Args:
        data_cluster (pd.DataFrame): DataFrame containing 'source_label', 'target_label', and 'barcode'.

    Returns:
        links (list): List of links for the Sankey chart.
        nodes (list): List of nodes for the Sankey chart.
    """
    try:
        # Validate input
        if not isinstance(data_cluster, pd.DataFrame):
            raise ValueError("data_cluster must be a pandas DataFrame.")
        if not all(col in data_cluster.columns for col in ['source_label', 'target_label', 'barcode']):
            raise ValueError("data_cluster must contain columns 'source_label', 'target_label', and 'barcode'.")

        links = []
        for i, j, k in zip(data_cluster['source_label'], data_cluster['target_label'], data_cluster['barcode']):
            if k > 0:
                links.append({"source": i, "target": j, "value": k})

        nodes = []
        for link in links:
            if link['source'] not in nodes:
                nodes.append(link['source'])
            if link['target'] not in nodes:
                nodes.append(link['target'])

        nodes = [{"name": node} for node in nodes]

        return links, nodes
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None
   


def plot_sankey_chart(nodes, links, chart_title="Sankey Diagram", chart_width="2000px", chart_height="2000px", theme='westeros'):
    """
    Plot a Sankey chart based on the provided nodes and links.

    Args:
        nodes (list): List of nodes for the Sankey chart.
        links (list): List of links for the Sankey chart.
        chart_title (str): Title of the Sankey chart.
        chart_width (str): Width of the Sankey chart.
        chart_height (str): Height of the Sankey chart.
        theme (str): Theme to be used for the Sankey chart.

    Returns:
        Sankey: Sankey chart object ready for rendering.
    """
    try:
        # Validate input
        if not isinstance(nodes, list) or not isinstance(links, list):
            raise ValueError("Nodes and links must be provided as lists.")

        c = (
            Sankey(init_opts=opts.InitOpts(width=chart_width, height=chart_height, theme=theme))
            .add(
                "",
                nodes=nodes,
                links=links,
                linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source"),
                label_opts=opts.LabelOpts(position="right"),
            )
            .set_global_opts(title_opts=opts.TitleOpts(title=chart_title))
            .render("image.html")
        )

        return c
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
