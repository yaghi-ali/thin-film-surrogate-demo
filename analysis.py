import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def reflectance(wavelength_nm,thickness_nm,n0=1.,n1=1.38,n2=1.52):
    w,d=np.asarray(wavelength_nm,float),np.asarray(thickness_nm,float)
    if np.any(~np.isfinite(w)) or np.any(~np.isfinite(d)) or np.any(w<=0) or np.any(d<0) or min(n0,n1,n2)<=0:
        raise ValueError('Require positive wavelengths/indices and nonnegative thickness.')
    r01=(n0-n1)/(n0+n1);r12=(n1-n2)/(n1+n2);phase=np.exp(4j*np.pi*n1*d/w)
    return np.abs((r01+r12*phase)/(1+r01*r12*phase))**2

def features(x):
    x=np.asarray(x,float)
    if x.ndim!=2 or x.shape[1]!=2 or not np.all(np.isfinite(x)):
        raise ValueError('Require finite N x 2 wavelength/thickness inputs.')
    z=(x-np.array([600,110]))/np.array([150,40])
    return np.column_stack([z[:,0]**i*z[:,1]**j for total in range(9) for i in range(total+1) for j in [total-i]])

def train(x,y,alpha=1e-6):
    f=features(x);y=np.asarray(y,float)
    if y.shape!=(len(x),) or not np.all(np.isfinite(y)) or alpha<=0:raise ValueError('Invalid targets or regularization.')
    mean=f[:,1:].mean(axis=0);scale=f[:,1:].std(axis=0);scale=np.maximum(scale,1e-12)
    f[:,1:]=(f[:,1:]-mean)/scale
    penalty=np.eye(f.shape[1])*alpha;penalty[0,0]=0
    coef=np.linalg.solve(f.T@f+penalty,f.T@y)
    return mean,scale,coef

def predict(x,state):
    mean,scale,coef=state;f=features(x);f[:,1:]=(f[:,1:]-mean)/scale;return f@coef

def run(out,seed=42):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(seed)
    xtrain=rng.uniform([450,70],[750,150],(1000,2));xtest=rng.uniform([450,70],[750,150],(400,2))
    ytrain=reflectance(xtrain[:,0],xtrain[:,1]);ytest=reflectance(xtest[:,0],xtest[:,1]);state=train(xtrain,ytrain);yp=predict(xtest,state)
    metrics={'seed':seed,'data':'synthetic analytical model','train_samples':1000,'held_out_samples':400,'test_rmse':float(np.sqrt(np.mean((yp-ytest)**2))),'mean_baseline_rmse':float(np.sqrt(np.mean((ytrain.mean()-ytest)**2))),'max_absolute_test_error':float(np.max(np.abs(yp-ytest))),'out_of_physical_range_count':int(np.sum((yp<0)|(yp>1)))}
    np.savetxt(out/'test_predictions.csv',np.column_stack([xtest,ytest,yp]),delimiter=',',header='wavelength_nm,thickness_nm,exact_reflectance,predicted_reflectance',comments='')
    np.savez(out/'model.npz',mean=state[0],scale=state[1],coef=state[2])
    fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].scatter(ytest,yp,s=7);ax[0].plot([ytest.min(),ytest.max()],[ytest.min(),ytest.max()],'k--');ax[0].set(xlabel='Exact reflectance',ylabel='Predicted reflectance',title='Held-out predictions')
    im=ax[1].scatter(xtest[:,0],xtest[:,1],c=np.abs(yp-ytest),s=12);fig.colorbar(im,ax=ax[1],label='Absolute reflectance error');ax[1].set(xlabel='Wavelength (nm)',ylabel='Thickness (nm)',title='Test-domain error')
    fig.tight_layout();fig.savefig(out/'preview.png',dpi=160);plt.close(fig);(out/'metrics.json').write_text(json.dumps(metrics,indent=2));return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results');p.add_argument('--seed',type=int,default=42);a=p.parse_args();print(json.dumps(run(a.out,a.seed),indent=2))
