#!/usr/bin/env python

#prior versions were messyy...
#v4 is an attempt to substantially clean this up
#note in prior versions also tried "log bins" for ranks, see v3 for that...
#v5 add parallel fitting to this - Nov21 for paper revisions
#1/15/2026 added ccc and fixed bug in KS (thanks Beatrix)
#1/22/2026 fixed parameter ranges to be more systematic

import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
from joblib import Parallel, delayed # type: ignore
import scipy.stats as st # type: ignore

#fixed parameter ranges for fitting grid search
almin=0.1; almax=3
al1min=0.4; al1max=5
al2min=0.1; al2max=1
phi_min=np.log10(1); phi_max=np.log10(1000)

#make into rank abundance with proper sorting / chopping etc
def make_ra(counts):
   
    l=counts[counts>0] #just keep nonzero
    R=len(l);  
    r=np.arange(R)+1
    a=-np.sort(-l)
    N=np.sum(a)
    pa=a/N
    cpa=np.cumsum(pa)

    return r,a,N,pa,cpa

#multinomial sampler for rank abundance
def resample_ra(ss,counts):
    
    r,a,N,pa0,cpa = make_ra(counts)

    rs = np.random.multinomial(n=ss,pvals=pa0) #resampled abundance

    rf,af,N,paf,cpa = make_ra(rs)
    
    return rf,af,paf

#ecology calculations from resampled list of abundances
def calc_ecol(counts):
    
    r,a,N,pa,cpa = make_ra(counts)

    R=max(r)
    D1=np.exp(-np.sum(pa*np.log(pa))) #evenness
    D2=1/np.sum(pa**2) #inv dominance
    
    #heuristic metrics
    maxclone=a[0]/N #proportional abundance of largest clonotype
    top10clones=np.sum(a[:10])/N #proportional abundance of top 10 clonotypes
    clonalf=np.sum(a[a>1])/N #fraction nonsingleton
    #nonsingler=max(r[a>1])
    
    return r,a,pa,cpa,R,D1,D2,N,R/N,D1/N,D2/N,maxclone,top10clones,clonalf

#compute probability distributions for single and double powerlaws
def pwl_fun_R(dist_name,xx):
    
    #single power law model
    if 'pwl1' in dist_name:
        al1,Rl=xx
        r = np.arange(10**Rl)+1
        a = r**-al1

    #double power law model - phi on linear scale (tried log too)
    if 'pwl2' in dist_name:
        al1,al2,phi,Rl=xx
        r = np.arange(10**Rl)+1
        a = r**-al1 + 1/phi * r**-al2

    #double power law model - put in terms of rcp (rank change point)
    # rcp^-al1 = phi * rcp^-al2
    # -al1 * ln rcp = ln (phi * rcp^-al2) = ln phi + -al2 * ln rcp
    # (al2-al1) * ln rcp = ln phi 
    # phi = rcp^(al2-al1)
    if dist_name=='pwl2_rcp':
        al1,al2,rcp,Rl=xx
        r = np.arange(10**Rl)+1
        phi = rcp**(al2-al1)
        a = r**-al1 + phi * r**-al2

    pa_model = a/np.sum(a) #normalize

    return pa_model

# max rank to fit up to
#fitting wasn't converging for too many ranks, because it cared about tail too much? so trying a bunch of functions
def calc_maxr(maxrchoice,a_data):
    
    #look at all the clones
    if maxrchoice == 'all':
        maxr=len(a_data) #assume start here
        
    #only look at the top 100 clones but correct in case there aren't even 100 ranks 
    elif maxrchoice == 'top100':
        maxr=np.min([100,len(a_data)]) 

    #fit to non singletons if there are any?
    elif maxrchoice == 'nonsingle':
        #make sure there are some nonsingletons
        if len(a_data[a_data>1])>0:
            maxr=len(a_data[a_data>1]) 
        else:
            maxr=len(a_data)
                
    else:
        print('input:',maxrchoice,'did not match calc_maxr options: all, top100, or nonsingle')
            
    return maxr
    
# objective function definitions, calculates error
# peforms the "trimming" of data up to maxr
def calc_err(model_properties,pa_model,a_data):
        
    xx, ss, dist_name, scorefun, maxrchoice, samplechoice = model_properties

    pa_data = a_data/sum(a_data) #normalize

    #make them only as long as the shorter one
    minr = np.min([len(pa_model),len(a_data)]) 
    pa_data = pa_data[:minr]
    pa_model = pa_model[:minr]
              
    maxr = calc_maxr(maxrchoice,a_data) #choose the maximum based on our definitions and the data

    #trim both using maxr
    pa_model_trimmed = pa_model[:maxr]
    pa_data_trimmed = pa_data[:maxr]

    #'RMS', 'logRMS', 'cRMS', 'KS', 'analytical'

    #simple RMS of pa
    if scorefun == 'RMS':
        resids = pa_model_trimmed - pa_data_trimmed
        err = np.sqrt(np.mean(resids**2))

    #RMS after taking logs
    elif scorefun == 'logRMS':
        resids = np.log(pa_model_trimmed) - np.log(pa_data_trimmed)
        err = np.sqrt(np.mean(resids**2))

    #RMS on cumulative pa
    elif scorefun == 'cRMS':
        resids = np.cumsum(pa_model_trimmed)-np.cumsum(pa_data_trimmed)
        err = np.sqrt(np.mean(resids**2))

    #kolmogorov smirnov statistic
    elif scorefun == 'KS':
        resids = np.cumsum(pa_model_trimmed)-np.cumsum(pa_data_trimmed)
        err = np.max(np.abs(resids))

    #analytical likelihood
    # ln L = \sum_i p_i(obs) * ln(p_i(model))
    # err is negative log likelihood because we are minimizing
    elif scorefun == 'analytical':
        likelihood = pa_data_trimmed * np.log(pa_model_trimmed)
        err = -np.sum(likelihood)
    
    else:
        print('input:',scorefun,'did not match calc_error options: RMS logRMS cRMS KS or analytical')
    
    return err
    
# calculate error for rank abundance, can vary maxr the objective functoin and how to "Sample"
def calc_sampled_err(model_properties, a_data):
    
    xx, ss, dist_name, scorefun, maxrchoice, samplechoice = model_properties
        
    pa_model0 = pwl_fun_R(dist_name, xx) #use functions to calculate
    
    #'exact', 'round', multinomial_10, multimin_10

    #no extra sampling, multinomial approximates true
    if samplechoice == 'exact':
        pa_model = pa_model0
        err = calc_err(model_properties,pa_model,a_data)
   
    #round the probabilitiles??
    elif samplechoice == 'round':
        pa_model = np.round(pa_model0*ss)
        pa_model = pa_model[pa_model>0] #get rid of zeros
        err = calc_err(model_properties,pa_model,a_data)

    #boostrap over multinomial samples
    elif 'multinomial' in samplechoice:
        num_replicates = int(samplechoice.split('_')[1]) #could pass this?
        err=0
        for i in range(num_replicates):
            ri,ai,pai=resample_ra(ss,pa_model0)
            err += calc_err(model_properties,pai,a_data)
        err = err/num_replicates

    #boostrap over multinomial samples - take the ABSOLUTE best fit (min)
    elif 'multimin' in samplechoice:
        num_replicates = int(samplechoice.split('_')[1]) #could pass this?
        ri,ai,pai=resample_ra(ss,pa_model0)
        err_best = calc_err(model_properties,pai,a_data)
        #now try num_replicates others to see if any are better
        for i in range(num_replicates-1):
            ri,ai,pai=resample_ra(ss,pa_model0)
            err_try = calc_err(model_properties,pai,a_data)
            if err_try < err_best:
                err_best = err_try
        err = err_best

    else:
        print('input:',samplechoice,'did not match fit_error options: exact round or multinomial')
        
    return err
        
        
#define my own minimizer, now using parallel always??
#in this version you loop to make the long vector of params and then operate on that again
def gsearch_fitp(model_properties, a_data):
    
    ss, model, objf, rmax, objs, gsize, nfits = model_properties

    mpl=[] #list for the estimated parameters
        
    if 'pwl1' in model:
        FQ = int(model.split('_')[1]) #could pass this?
        for al in np.linspace(almin,almax,gsize):
            xx = al,FQ
            mpl.append([xx, ss, model, objf, rmax, objs])
    
    if 'pwl2' in model:
        FR = int(model.split('_')[1]) #could pass this?
        for al1 in np.linspace(al1min,al1max,gsize):
            for al2 in np.linspace(al2min,al2max,gsize):
                for phi in np.logspace(phi_min,phi_max,gsize):
                    xx = al1,al2,phi,FR
                    mpl.append([xx, ss, model, objf, rmax, objs])

    #nwo run for whole model_param_l in parallel
    errp = Parallel(n_jobs=16)(delayed(calc_sampled_err)(mi, a_data)for mi in mpl)
        
    #now do the loops to append errors in
    fl=[] #full grid search list
    for i,mi in enumerate(mpl):
        fl.append(list(mi[0])+[errp[i]])
 
    #sort and output df
    if 'pwl1' in model:
        dff = pd.DataFrame(fl,columns = ['Fal','FQ','err'])

    if 'pwl2' in model:
        dff = pd.DataFrame(fl,columns = ['Fal1','Fal2','Fphi','FR','err'])
 
    dfs=dff.sort_values(by='err') #sort fitted data frame results
    
    best_params = dfs.iloc[0:nfits].mean().values #can average over some range

    return dfs, best_params #note size depends on pwl1 pwl2

#function to compute ccc
def concordance_corr(x, y):

    mean_x = np.mean(x); var_x = np.var(x, ddof=1);  
    mean_y = np.mean(y); var_y = np.var(y, ddof=1)
    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    
    return 2 * cov_xy / (var_x + var_y + (mean_x - mean_y)**2)

#function to make an lhs sample, useful for testing
def genLHS(param_ranges,n_samples):
    
    # Generate LHS in unit cube
    sampler = st.qmc.LatinHypercube(d=len(param_ranges))
    lhs_unit = sampler.random(n=n_samples)

    # Scale to actual ranges
    bounds = np.array(param_ranges)
    lhs_scaled = st.qmc.scale(lhs_unit, bounds[:, 0], bounds[:, 1])
    
    return lhs_scaled
