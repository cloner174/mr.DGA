import numpy as np
from scipy import stats

def sublist(List, n):
        
        li = List
        len_ = len(li)
        max_index = ( len_ - 1 )
        splitCoef = int( round(  ( len_ / n ), 0 ) )
        split_index = list( ( i for i in range(0, max_index, splitCoef) ) )
        
        ll = []
        for i in range( len(split_index)):
            te = split_index[i]
            if i == 0:
                sub1 = li[:te+1]
                if len(sub1) >= 2:
                  ll.append(li[:te+1])
            else:
               if i == len(split_index)-1:
                   ll.append( li[te:] )
               elif te < max_index:
                   ll.append( ( li[  split_index[i-1] : split_index[i]  ]  ) )
               else:
                   pass
        
        return ll
    
    
def stat(subset, quantiles_ = None):
        #Q = input( " Please Inter the prob Number for quantiles ...... Leave it blank for defult .. \n   You may inter here .....--->>>>.....")
        #if Q == '':
        #    quantiles = [0.2, 0.25, 0.27, 0.3, 0.33, 0.35, 0.36, 0.60, 0.61, 0.64, 0.66, 0.69, 0.72, 0.75, 0.95]
        #elif Q ==  ' ':
        #    quantiles = [0.2, 0.25, 0.27, 0.3, 0.33, 0.35, 0.36, 0.60, 0.61, 0.64, 0.66, 0.69, 0.72, 0.75, 0.95]
        #else:
        #    try:
        #        quantiles = list(json.loads(Q))
        #    except:
        #        raise ValueError( " Sorry , Something is not right ! ")
        if quantiles_:
            quantiles = quantiles_
        else:
            #quantiles = [0.5, 0.1, 0.2, 0.3, 0.35, 0.40, 0.42, 0.44, 0.48, 0.52, 0.57, 0.65, 0.70, 0.75, 0.85, 0.95]
            quantiles = [0.25, 0.75]

        
        return [
            np.mean(subset),
            np.median(subset),
            stats.mode(subset)[0],
            np.std(subset),
            np.var(subset),
            *[np.quantile(subset, q) for q in quantiles]
        ]

