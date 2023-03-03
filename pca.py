import os
from os.path import join

import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


if __name__ == '__main__':
    embed_dir = '/Users/minh/Documents/2022/smart_contract/mando/original-ge-sc/experiments/ge-sc-data/source_code/java/Neo4j/cfg'
    embed_files = [f for f in os.listdir(embed_dir) if f.endswith('.pt')]
    gt_file = '/Users/minh/Documents/2022/smart_contract/mando/original-ge-sc/experiments/ge-sc-data/source_code/java/Neo4j/cfg/gt.gpickle'
    gt = torch.load(gt_file)
    data = None
    for embed in embed_files:
        e = torch.load(join(embed_dir, embed))
        data = e if data is None else torch.cat((data, e))
    data = data.detach().numpy()
    pca = PCA(n_components=2)
    X_r = pca.fit(data).transform(data)
    print("explained variance ratio (first two components): %s"% str(pca.explained_variance_ratio_))
    plt.figure()
    colors = ["navy", "turquoise",]
    lw = 2

    for color, i, target_name in zip(colors, [0, 1], ['bug', 'normal']):
        plt.scatter(
            X_r[gt == i, 0], X_r[gt == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of Neo4j java dataset")
    plt.show()
