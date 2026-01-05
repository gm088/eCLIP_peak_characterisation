import numpy as np
import matplotlib.pyplot as plt

def plot_feats(peaks, labels, fpath= None, bins=10):
	
    fig = plt.figure(figsize=(10,6))
    
    for i in range(peaks.shape[1]):
        plt.subplot(5,5,i+1)
        plt.hist( peaks.iloc[:,i] , bins=bins)
        if labels:
			#plt.xlabel("{0}".format(labels[i]))
            plt.annotate("{0}".format(labels[i]), xy=(0.05,0.8), xycoords='axes fraction')
		    
    #fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.show()
    if fpath:
        fig.savefig(fpath, bbox_inches='tight')

def plot_feats_split(peaks, reqdclus, fpath=None, bins=10, labels=None):
	
    fig = plt.figure(figsize=(12,10))
    p1 = peaks.loc[peaks['cluster'] == reqdclus]
    p2 = peaks.loc[peaks['cluster'] != reqdclus]
    
    ### just split into 2 hists
    for i in range(peaks.shape[1]):
        plt.subplot(5,5,i+1)
        plt.hist( p2.iloc[:,i] , bins=bins)
        plt.hist( p1.iloc[:,i] , bins=bins, color="red")
        if labels:
			#plt.xlabel("{0}".format(labels[i]))
            plt.annotate("{0}".format(labels[i]), xy=(0.05,0.8), xycoords='axes fraction')
		    
    #fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.show()
    if fpath:
        fig.savefig(fpath, bbox_inches='tight')

def plot_UMAPs_3D(projections, peaks, labs, e, a, r, fpath=None):
	
    fig = plt.figure(figsize=(15,15))
    
    for i in range(peaks.shape[1]):
        ax = fig.add_subplot(4,4,i+1, projection='3d')

        ax.scatter(projections[:, 0], projections[:, 1], projections[:, 2], c=peaks[:,i], s=0.5, cmap='magma')
        ax.set_title("{0}".format(labs[i]))
        
    for i in fig.axes:
            i.view_init(elev=e, azim=a, roll=r)
		    
    fig.patch.set_facecolor('white')
    plt.show()
    if fpath:
        fig.savefig(fpath, bbox_inches='tight')


def log2Transform(vals, pseudocount = 0):
    return np.log2(vals + pseudocount)

def integerTransform(vals):
    return vals.astype(int)

def noiseAddedLog10Transform(vals_pdSeries):
    ## note that this needs to be a pd.Series
    ## this is for strictly non-negative distributions
    noise = np.random.normal(0, np.mean(np.diff(vals_pdSeries.value_counts().axes))/2, len(vals_pdSeries))
    new_vals = vals_pdSeries + noise
    new_transformed = np.log10( np.abs(new_vals) ) # negative values not meaningful

    return(new_transformed)

###############


def feat_correlations(peaks, labels=None):
	
    fig = plt.figure(figsize=(15,15))
    nfeats = peaks.shape[1]
    correlations = peaks.corr()

    for i in range(nfeats):
        for j in range(nfeats):
            plt.subplot(nfeats, nfeats, (i)*nfeats + (j+1) )
            plt.scatter( peaks.iloc[:,j], peaks.iloc[:,i] )
            if labels:
                #plt.ylabel("{0}".format(labels[j]))
                #plt.xlabel("{0}".format(labels[i]))
                plt.annotate("{0}:{1}".format(labels[i], labels[j]), xy=(0.05,0.8), xycoords='axes fraction')
                plt.annotate("r2 = {:.3f}".format(correlations.iloc[i,j]), xy=(0.15,0.1), xycoords='axes fraction')
		    
    fig.patch.set_facecolor('white')
    #plt.tight_layout()
    plt.show()

def plot_UMAPs(projections, peaks, quantiles):
	
    fig = plt.figure(figsize=(15,15))
    
    for i in range(peaks.shape[1]):
        plt.subplot(4,4,i+1)
        
        cond = peaks.iloc[:,i] > np.quantile(peaks.iloc[:,i], q=quantiles[i])
        
        for color, cat, lab in zip(['green', 'orange'], [cond, ~cond], ['more', 'less']):
            plt.scatter(projections[cat, 0], projections[cat, 1],
            c=color, label=lab, s=0.5, edgecolors='none')
        plt.legend()

        plt.title("{0}, pctile={1}".format(peaks.columns[i], quantiles[i]))
		    
    fig.patch.set_facecolor('white')
    plt.show()

def plot_UMAPs_2(projections, peaks, labs):
	
    fig = plt.figure(figsize=(15,15))
    
    for i in range(peaks.shape[1]):
        plt.subplot(4,4,i+1)

        plt.scatter(projections[:, 0], projections[:, 1], c=peaks[:,i], s=0.5)

        plt.title("{0}".format(labs[i]))
		    
    fig.patch.set_facecolor('white')
    plt.show()

