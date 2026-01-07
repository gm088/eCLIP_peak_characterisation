import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
import umap
import sklearn.cluster as cluster
from util_funcs import *
import os

### peaks are imported, then transformed, then scaled, then UMAPped

class eCLIP_peaks(object):

    def __init__(self, data):
        self.rawdata = data
        self.scaled_data = None
        self.umap_projection = None
        self.clusts = None

    def transform_data(self, col_ids = None, transforms = None):
        ##transform some of the data columns
        if not col_ids and not transforms:
            col_ids = ['rpkm', 'coding', 'regulated', 'dist_to_TSS', 'numpeaks_per_clus']
            transforms = [log2Transform, integerTransform, noiseAddedLog10Transform, noiseAddedLog10Transform, noiseAddedLog10Transform]
        
        for col, transform in zip(col_ids, transforms):
            ### pass the column to be transformed, and the transform function
            self.rawdata[col] = transform(self.rawdata[col])

    def scale_data(self):
        ## construct sklearn pipeline using scaler objs and assign scaled data to attribute
        peaks_train = self.rawdata.drop('coding', axis=1)
        pipe = Pipeline([ ( 'scaler', sklearn.preprocessing.RobustScaler() ), ( 'power_transform', sklearn.preprocessing.PowerTransformer() ) ])
        self.scaled_data = pipe.fit_transform(peaks_train)

    def run_UMAP(self, UMAP_inst):
        ## initialise the UMAP and fit, train and transform, assign to attribute
        manifold = UMAP_inst.fit(self.scaled_data)
        self.umap_projection = manifold.fit_transform(self.scaled_data)

    def kmeans(self, kmeans_instance):
        self.clusts = kmeans_instance.fit_predict(self.umap_projection)


#####

##read the file
peaks = pd.read_table("/hpcnfs/data/GN2/gmandana/bin/4.1.0/home/ieo5559/ENHANCEDCLIP/characterisation_2/relax_peak_region_features.dat")
#for now ignore the kmers
peaks = peaks.drop(peaks.filter(regex=".*mer.*|tx").columns, axis=1)
#set NaN to zero
peaks[peaks.isna()] = 0

## set seed for consistency - use for UMAP and kmeans
randomseed = 42
number_of_clusters = 4
outdir = "/hpcnfs/data/GN2/gmandana/bin/eCLIP_peak_characterisation/5Jan26"
try:
    os.mkdir(outdir)
except FileExistsError:
    pass

#### I N S T A N T I A T E
myPeaks = eCLIP_peaks(peaks)
data_columns = list( myPeaks.rawdata.columns )

### apply transformations
myPeaks.transform_data()
plot_feats(pd.DataFrame(myPeaks.rawdata), labels = data_columns, bins=50, fpath = os.path.join(outdir, "feature_transformed.pdf"))

print("features transformed")

## scale the transformed data
myPeaks.scale_data()
plot_feats(pd.DataFrame(myPeaks.scaled_data), labels = data_columns[1:], bins=50, fpath = os.path.join(outdir, "feature_posttransform_scaling.pdf"))

## run UMAP and clustering
myPeaks.run_UMAP( umap.UMAP(n_neighbors = 50, min_dist=0.0, n_components=3, random_state=randomseed) )
myPeaks.kmeans( cluster.KMeans(n_clusters=number_of_clusters, random_state=randomseed) )

####### plotting ##########
fig = plt.figure(figsize= (10,10))
ax = fig.add_subplot(projection='3d')

for color, cat, lab in zip(['red', 'blue'], [1, 0], ['coding', 'noncoding']):
    ax.scatter(myPeaks.umap_projection[peaks.coding == cat, 0], myPeaks.umap_projection[peaks.coding == cat, 1], myPeaks.umap_projection[peaks.coding == cat, 2],
    c=color, label=lab, s=0.5)
ax.view_init(elev=70, azim=-70, roll=0)

fig.savefig( os.path.join(outdir, "coding_noncoding.pdf") , bbox_inches='tight')

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection='3d')
scatter = ax.scatter(myPeaks.umap_projection[:, 0], myPeaks.umap_projection[:, 1], myPeaks.umap_projection[:,2 ], c=myPeaks.clusts, s=0.5, cmap='Spectral')
ax.view_init(elev=70, azim=-70, roll=0)

legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
ax.add_artist(legend1)

fig.savefig(os.path.join(outdir, "clusters.pdf") , bbox_inches='tight')

plot_UMAPs_3D(myPeaks.umap_projection, peaks=myPeaks.scaled_data, labs=data_columns[1:], e=70, a=-70, r=0, fpath = os.path.join(outdir, "feature_vals.pdf"))

######### lastly, append cluster assignments and write to file

peaks_scaled_df = pd.DataFrame(myPeaks.scaled_data, columns=data_columns[1:])
peaks_scaled_df.insert(0, "cluster", myPeaks.clusts)

# visualise feat distributions
for i in range(1, number_of_clusters+1):
    plot_feats_split(peaks=peaks_scaled_df, reqdclus = i-1, labels = list(peaks_scaled_df.columns), bins=50, fpath = os.path.join( outdir, """feature_dist_clus_{0}.pdf""".format(i) ) )

peaks.insert(1, "cluster", myPeaks.clusts)
peaks.to_csv( os.path.join(outdir, "peaks_clustered.dat") , sep = "\t")




