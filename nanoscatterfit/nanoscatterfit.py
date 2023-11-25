# -*- coding: utf-8 -*-
"""Main module."""

#%%

import pandas as pd

from typing import Tuple, Optional, Union,Dict,Any,List
from scipy import interpolate, sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from lmfit import Minimizer, Parameters
from lmfit.minimizer import MinimizerResult
from lmfit.lineshapes import gaussian, lognormal


import os
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt



from .utils import *
from .dataimport import *

    


@log_function_call
def calc_structurefactor(sampledata: pd.DataFrame, suspdata: pd.DataFrame, normalize:bool=False)-> pd.DataFrame:
    """
    Calculates the structure factor and adds it as column to the sampledata 

    Parameters:
    - sampledata : pd.DataFrame
        Sample data containing at least a 'q' and 'I' column.
    - suspdata : pd.DataFrame
        Suspension data containing at least a 'q' and 'I' column.
    - normalize : bool, optional
        normalizes area underneath the 'q' and 'S' columns to 1 if True. Default is False.
    Returns:
    - sampledata: pd.DataFrame
        dataframe of sample augmented by the form factor and structure factor column.
    """
    logging.info('Calculating structure factor')

    if 'q' not in sampledata.columns or 'I' not in sampledata.columns:
        logging.error("Sample data is missing required columns.")
        return

    if 'q' not in suspdata.columns or 'I' not in suspdata.columns:
        logging.error("Suspension data is missing required columns.")
        return

    f = interpolate.interp1d(suspdata['q'], suspdata['I'], kind='cubic')
    if (sampledata['q'].iloc[0] < suspdata['q'].iloc[0]) | (sampledata['q'].iloc[-1] < suspdata['q'].iloc[-1]):
        logging.warning("Sample data starts or ends before suspension data. sampledata will be cut.")
        sampledata= cut_diffractogram(sampledata,suspdata['q'].iloc[0],suspdata['q'].iloc[-1])
    sampledata['F'] = pd.Series(f(sampledata['q']))
    sampledata['S'] = sampledata['I'] / sampledata['F']
    if normalize==True:
        sampledata['S'] = normalize_area(sampledata['q'], sampledata['S'])
    return sampledata
@log_function_call
def baseline_als_optimized(y, lam, p, niter=50):
    ''''
    baseline ALS fit which works quite well for spectroscopic and diffraction data
    '''
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z



@log_function_call    
def auto_baseline_optimizer(y:pd.Series,
                            alpha:int=1,beta:int=15,
                            lam:Tuple[int,int]=(-5, -2),
                            p:Tuple[int,int]=(-9, -5)
                            ) -> Optional[np.ndarray]:
    """
    Automatically optimizes the baseline of the diffractogram using the cost function defined in the cost_func function.
    It is important to specify the range of hyperparameters lam and p to search for the optimal baseline.
    Parameters:
    - y : pd.Series
        Series containing the diffractogram data.
    - alpha : int, optional
        Weight of the term 1 in the cost function. Default is 2.
    - beta : int, optional
        Weight of the term 2 in the cost function. Default is 10.
    - lam : tuple of int, optional
        Range of lambda values to search for the optimal baseline. Default is (-5, -2).
    - p : tuple of int, optional
        Range of p values to search for the optimal baseline. Default is (-9, -5).

    Returns:
    - np.array or None
        Optimal baseline, or None if the input series is empty or invalid.
    """
    if y.empty:
        logging.warning("Input series is empty.")
        return None


    def filter_monotonic_peaks(y, peaks):
        """Keep only peaks that belong to a monotonically increasing sequence."""
        filtered_peaks = [peaks[0]]
        for i in range(1, len(peaks)):
            if y[peaks[i]] >= y[filtered_peaks[-1]]:
                filtered_peaks.append(peaks[i])
            elif y[peaks[i]] <= y[filtered_peaks[-1]]:
                filtered_peaks[-1]=peaks[i]
                # print('erase last peak')
        return np.array(filtered_peaks)
    # Define a cost function to evaluate the baseline
    def cost_func(y, baseline, alpha=alpha, beta=beta):
        '''
        This is a cost function which tries to fit a basaeline which (Term 1) decreases the distance to the strongest minima and (Term 2) keeps the curvature positive to prevent overfitting the whole diffractogram
        '''
        # Find the strong minima in the data
        inverted_y = -y  # Invert the data to find minima as peaks
        peaks, _ = find_peaks(inverted_y, distance=3)  # Replace 0.5 with an appropriate threshold for your data
        # print(peaks)
        # Filter peaks to get a monotonically increasing sequence
        # filtered_peaks = filter_monotonic_peaks(y, peaks) #!didnt work well
        filtered_peaks=peaks
        # print(filtered_peaks)
        # Term 1: Distance from strong minima (we consider only the found peaks)
        term1 = np.sum(np.abs(y[filtered_peaks] - baseline[filtered_peaks]))

        # # Term 2a: Curvature of the baseline 
        #* didnt work well
        curvature = np.diff(baseline, 2)
        # term2 = np.sum(np.abs(curvature))
        
        # Term 2: Penalty for negative curvature
        term3 = np.sum(np.clip(-curvature, 0, None))

        return alpha * term1  + beta * term3




    # Grid search
    best_cost = float('inf')
    best_lam = None
    best_p = None
    best_baseline = None
    for _lam in np.logspace(lam[0], lam[1], lam[1]-lam[0]+1):
        for _p in np.logspace(p[0], p[1], p[1]-p[0]+1):
            baseline = baseline_als_optimized(y, _lam, _p)
            cost = cost_func(y, baseline)
            if cost < best_cost:
                best_cost = cost
                best_lam = _lam
                best_p = _p
                best_baseline = baseline

    logging.info(f"Best lambda: {best_lam}, Best p: {best_p}") 
    return best_baseline





def structuremodel(pars:Parameters, x:pd.Series, y:pd.Series, n_gaussians:int)->pd.Series:
    """Residual function which is used to fit several gauss functions to the data x,y

    Parameters
    ----------
    - pars : Parameters
        given parameters of the model
    - x : pd.Series
        x values of data
    - y : pd.Series
        y values of data
    - n_gaussians : int
        number of gaussian functions which will be used

    Returns
    -------
    - pd.Series
        returns the residual of fitted model and the actual y values
    """
    model = np.zeros_like(x)
    for i in range(1, n_gaussians + 1):
        amp_key = f'amp{i}'
        cen_key = f'cen{i}'
        sigma_key = f'sigma{i}'
        model += gaussian(x, pars[amp_key], pars[cen_key], pars[sigma_key])
    model += pars['y0']
    return model - y
@log_function_call
def fit_structurefactor(x:pd.Series,y:pd.Series,
                      structure:str,
                      sigmarange:Tuple=(0,0.05),
                      max_sigma_diff:int=None,
                      q_range: Union[str,Tuple[float,float]] ='auto',
                      q111:Union[str,float] ='auto',
                      )->MinimizerResult:
    """fits the Structure factor with a model depending on the crystal structure

    Parameters
    ----------
    - x : pd.Series
        q values of the diffractogram
    - y : pd.Series
        structure factor of the diffractogram
    - structure : str
        possibilities: 'fcc'
    - sigmarange : Tuple, optional
        constrain for the min and max possiblee peak width, by default (0,0.05)
    - max_sigma_diff : int, optional
        e.g. 0.1. optional parameter to constrain the maximum relative difference between the peaks, if, by default None
    - q_range : Union[str,Tuple[float,float]], optional
        if 'auto' the function will cut the diffractogram to the first and last tale found, else q range must be provided as tuple, by default 'auto'
    - q111 : Union[str,float], optional
        estimated q value of the (111) peak. if 'auto' is used, the algorithm is using the first peak in the diffractorgram, by default 'auto'
    Returns
    -------
    - Tuple[pd.Series,MinimizerResult]
        overall fit and MinimizerResult class containing the fit parameters, such as peak widths and positions
    """
    if q111=='auto' :
        peaks=find_peaks(y, prominence=0.1)
        if len(peaks)==0:
            logging.info(f'found {len(peaks)} peaks in y {y}')
        q111=x.iloc[peaks[0][0]]*0.985 #estimate for the first position of q
        logging.info(f'guess of the (111) peak position is q = {q111}')
    
    #determine max q depending on the amount of peaks to fit, adjust 
    peakpositions=indcate_diffractionpattern(q111,structure=structure,ax=None)
    logging.info(f'assumed peaks at positions: {peakpositions}')
    if q_range == 'auto':
        x,y = auto_cutdiffr(x,y,peakpositions[-1])
        logging.info(f'automatically cut the diffractorgram to the first and last tale found')
        
    elif type(q_range)==tuple:
        df=pd.DataFrame([x,y])
        df=cut_diffractogram(df,*q_range)
        x,y=df.iloc[:,0],df.iloc[:,1]

    else:
        logging.error(f'q_range must be tuple of floats or "auto", not {type(q_range)}')
    #initiate the restructions acc. to crystal structure
    pfit = Parameters()

    if structure=='fcc':
         ####number of gauss functions
        #in fcc the relative peak positions for the peak (j k l) is e.g. q111*(j + k + l)*(1/3)**(0.5)
        relativepositions=[3,4,8,11,12,16,19,20,24,27] #TODO adjust this later to just provide a list of possible signals depending of the structure
        n_gaussians=len(relativepositions)
        expression= '/3)**(0.5)'   
        mathexpression= lambda i,q: q*(i/3)**(0.5)   

        
        
    else:
        logging.info('unknown crystal structure given as input')
        return None

    
    for i in range(n_gaussians):
        if i==0:
            pfit.add(name='cen'+str(i+1), value=q111,min=q111*0.93,vary=True)
            pfit.add(name=f'sigma{i+1}', value=0.03, min=0, max=0.05)
        else:

            q=mathexpression(relativepositions[i],q111)
            logging.info(f'try adding gaussian no {i+1} at q = {q}')
            if q>x.iloc[-1]: 
                logging.info(f'q={q} is larger than the maximum q={x.iloc[-1]} - breaking')
                imax=i
                break # breaks the loop in order to not put more peaks as theoretically required
            pfit.add(name='cen'+str(i+1), expr='cen1*('+str(relativepositions[i])+expression)
            # Set initial value and constraints for sigma
            if max_sigma_diff is not None:
                # Add constraints to ensure sigma values are within max_sigma_diff of each other
                pfit.add(f'sigma_diff{i+1}', value=0, min=-max_sigma_diff, max=max_sigma_diff) 
                pfit.add(name=f'sigma{i+1}', expr=f'sigma1*(1+sigma_diff{i+1})')
    
            else:    
                pfit.add(name=f'sigma{i+1}', value=(sigmarange[1]-sigmarange[0])/2+sigmarange[0], min=sigmarange[0], max=sigmarange[1]) 
            
           
        pfit.add(name='amp'+str(i+1), value=0.03,min=0,max=0.1)
        # pfit.add(name='sigma'+str(i+1), value=0.03,min=0,max=0.1)

    pfit.add(name='y0',min=0, value=0,max=0.02)
    if not imax: imax=n_gaussians
    logging.info(f'number of gaussians={imax}')
    mini = Minimizer(structuremodel, pfit, fcn_args=(x, y, imax))
    res = mini.leastsq()
    res.rsquared=1 - res.residual.var() / np.var(y)  # needs to be added, as lmfit  has it not included. -> can be called using out.rsquared.
    logging.info(f'fit successfull with R2 = {res.rsquared:2f}')
    # best_fit = y + out.residual #optional but can also be calculated later if required
    return res 

@log_function_call
def indcate_diffractionpattern(q111:float,
                      structure:str,
                      ax=None,
                      )->List[float]:
    
    

    if structure=='fcc':

        #in fcc the relative peak positions for the peak (j k l) is e.g. q111*(j + k + l)*(1/3)**(0.5)
        relativepositions=[3,4,8,11,12,16,19,20,24,27] #TODO adjust this later to just provide a list of possible signals depending of the structure

        expression= lambda i,q: q*(i/3)**(0.5)   
        
    else:
        logging.info('unknown crystal structure given as input')
        return None
    

    peakpositions=[]
    for i in relativepositions:
        peakpositions.append(expression(i,q111))
        if ax is not None: ax.axvline(expression(i,q111),color='red',linewidth=1,alpha=0.5)
    return peakpositions

def get_structureinfo(out:MinimizerResult,
                      structure:str='fcc'
                      )->Tuple[float,float]:
    """calculates the lattice constant and domain size from the fitted parameters.
    The function returns a tuple containing the lattice constant and domain size, respectively.
    
    Parameters
    ----------
    -  out : MinimizerResult
        Fit result obtained using the fit_structurefactor() function.
    
    - structure : str, optional
        crystal structure, by default 'fcc'

    Returns
    -------
    - Tuple[float,float]
        lattice constant and domain size in nm
    """
    a=2*3**0.5*np.pi/out.params["cen1"].value
    fwhm=2*out.params["sigma1"].value*np.sqrt(2*np.log(2))
    Dsize=2*np.pi/fwhm
    return a, Dsize








# Define form factor functions
def spherical_form_factor(q, R):
    qr = q * R
    return (3 * (np.sin(qr) - qr * np.cos(qr)) / qr**3) ** 2

def cubic_form_factor(q, R):
    return np.sinc(q * R)**2

# Main scattering intensity function
@log_function_call
def formfactor(q, mu, sigma, C, y0,distribution, shape,distribution_type='volume' ):
    amp=1
    if distribution == 'lognormal':
        mu = np.log(mu)
        size_distribution = lognormal
    elif distribution == 'normal':
        sigma = sigma * mu
        size_distribution = gaussian

    if shape == 'spherical':
        form_factor = spherical_form_factor
    elif shape == 'cubic':
        form_factor = cubic_form_factor

    if distribution_type == 'volume':
        intensity_func = lambda R: 10**C * form_factor(q, R) * size_distribution(R,amp, mu, sigma) + y0
    elif distribution_type == 'number':
        intensity_func = lambda R: 10**C * form_factor(q, R) * size_distribution(R,amp, mu, sigma) * R**3  + y0
    else:
        raise ValueError("Invalid distribution type. Choose 'volume' or 'number'.")
    upper_limit = 10 * mu  # Example: 10 times the mean size
    lower_limit = 1e-3    # A small positive number to avoid division by zero issues
    return integrate.quad(intensity_func,  lower_limit, upper_limit)[0]

formfactor=np.vectorize(formfactor)
@log_function_call                                  
def fit_formfactor(q_data, I_data,initial_params, distribution='lognormal', shape='spherical', distribution_type='volume'):

    def model(q, mu, sigma, C, y0):
        # mu, sigma, C, y0=initial_params
        
        return formfactor(q,mu, sigma, C, y0,distribution, shape,distribution_type)
    logging.info(f'fitting formfactor model to SAXS data...')
    #* fit the log is required to fit optimally over several orders of magnitude
    popt, pcov = curve_fit(lambda q,mu, sigma, C, y0:np.log(model(q,mu, sigma, C, y0)), q_data, np.log(I_data), p0=initial_params)
    logging.info('fit successfull')
    if distribution=='normal':fitresult=f'mean size: {popt[0]:.2f} nm +/- {popt[1]*100:.1f} %'
    elif distribution=='lognormal':fitresult=f'mean size: {np.exp(np.log(popt[0])+popt[1]**2/2):.2f} nm, mode size: {np.exp(np.log(popt[0])-popt[1]**2):.2f} nm +/- {popt[1]*100:.1f} %'
    logging.info(fitresult)
    return popt, pcov, fitresult
@log_function_call 
def plot_formmodel(q_data,  params, distribution='lognormal', shape='spherical', distribution_type='volume', ax=None, color='red',label=None):
    def model(q, mu, sigma, C, y0):
        # mu, sigma, C, y0=initial_params
        return formfactor(q,mu, sigma, C, y0,distribution, shape,distribution_type)
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel('Scattering vector, q ($\mathregular{nm^{-1}}$)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_yscale('log')
    # ax.plot(q_data, I_data, 'o', label='data')
    p=ax.plot(q_data, model(q_data,*params),color=color, label=label)
    # ax.set_xlabel('Scattering vector, q ($\mathregular{nm^{-1}}$)')
    # ax.set_ylabel('Intensity (a.u.)')
    # ax.set_yscale('log')
    if label: ax.legend()
    return p


#* Data visualization

@log_function_call
def plot(df: pd.DataFrame, x: str = 'q', y: str = 'I', 
         subtractbaseline=False,
         ax: Optional[plt.Axes] = None, scale: str = 'log',
         line:bool=True,
         label: Optional[str] = None)->Tuple[matplotlib.lines.Line2D,Optional[matplotlib.lines.Line2D]]:
    """
    Plots the data based on the specified columns and settings.

    Parameters:
    - df : pd.DataFrame
        DataFrame containing the data to plot.
    - x : str, optional
        Column name for the x-axis. Default is 'q'.
    - y : str, optional
        Column name for the y-axis. Default is 'I'.
    - subtractbaseline : bool, optional
        Whether to subtract the baseline from the data. Only works when y='S' and a baseline is already created. Default is False.
        If a baseline is in the DataFrame, the baseline will be plotted when y='S'.
    - ax : matplotlib.pyplot.Axes, optional
        Axis on which to plot the data. If None, a new subplot is created.
    - scale : str, optional
        Scale for the y-axis. Default is 'log'.
    - line : bool, optional
        if False, function returns a scatterplot, else a 2DLine plot
    - label : str, optional
        Label for the curve on the plot.

    Returns:
    - Tuple[matplotlib.lines.Line2D, Optional[matplotlib.lines.Line2D, None]]
        Tuple containing the line object and the baseline line object.
        The baseline line object is None if the 'baseline' column is not present in the DataFrame.
    """

    # Check if essential columns exist in DataFrame
    if x not in df.columns or y not in df.columns:
        print(f"Error: DataFrame does not contain necessary columns {x} and/or {y}.")
        return

    # Create a new subplot if no axis is specified
    if ax is None:
        fig, ax = plt.subplots()
    
    if (subtractbaseline==True) & (y=='S'):
        if ('baseline' in df.columns):
            y_= df[y] - df['baseline']
        else: logging.warning(f"No baseline column found in DataFrame.")
    else: y_=df[y]
    
    # Plot the data
    if line==True:
        p=ax.plot(df[x], y_, label=label)
        p=p[0]
    else: p=ax.scatter(df[x], y_, label=label)
    if (y=='S') & ('baseline' in df.columns)&(subtractbaseline==False):
        b=ax.plot(df[x], df['baseline'],color='red', label='baseline')
    else: b=None
    # Set the y-axis scale
    ax.set_yscale(scale)

    # Set axis labels
    y_label_map = {'I': 'Intensity (a.u.)','F': 'Intensity (a.u.)', 'S': 'Structure Factor (a.u.)'}
    y_label = y_label_map.get(y, y)
    ax.set_xlabel('Scattering vector ($\mathregular{nm^{-1}}$)')
    ax.set_ylabel(y_label)

    # Add legend if label is specified
    if label:ax.legend()

    
    if b is not None:
        return p,b
    else:
        return p
     
@log_function_call
def plot_structuremodel(x_array:np.array,out:MinimizerResult, 
             peakcolor:str='green',fitcolor:str='black',
             ax:Optional[plt.Axes]=None,
             label: Optional[str] = None,
             yshift:int=0
             )->List[matplotlib.lines.Line2D]:
    """plots the fitted model using the parameters from the MinimizerResult class.
    The function returns a list of the plots, which can be used to further customize the plot.
    
    Parameters
    ----------
    - x_array : np.array
        data array for the x axis
    - out : MinimizerResult
        Fit results provided by the function fit_structurefactor()
    - peakcolor : str, optional
        color of the single gauss functions, by default 'green'
    - fitcolor : str, optional
        color of the overall fit, by default 'black'
    - ax : plt.Axes, optional
        in case of given axis shall be used, otherwise a new one is created., by default None
    - label: str, optional
        label for the overall fit, by default None
    Returns
    -------
    - List
        List of matplotlib.lines.Line2D and matplotlib.collections.PolyCollection containing the plots of gauss functions and area fillings, respectively. Allows for subsequent adjustments.
    """
    if ax is None: fig, ax =plt.subplots()

    y0=out.params['y0'].value
    # n_gaussians=int(list(out.params)[-2].replace('sigma',''))
    # logging.info(f'number of gaussians is {n_gaussians}')
    plots=[]
    gauss_peak_sum=np.array([y0]*len(x_array))

    logging.info(f'plotting gauss functions...')
    for i in range(len(out.params)):
        try:
            amp=out.params['amp'+str(i+1)].value
            cen=out.params['cen'+str(i+1)].value
            sigma=out.params['sigma'+str(i+1)].value
            gauss_peak = gaussian(x_array, amp, cen, sigma)
            gauss_peak_sum=gauss_peak_sum+gauss_peak
            peak_=ax.plot(x_array, gauss_peak+yshift, peakcolor)
            area_=ax.fill_between(x_array, gauss_peak.min()+yshift, gauss_peak+yshift, facecolor=peakcolor, alpha=0.1)
            plots.append((peak_,area_))
        except Exception as e:
            logging.info(f'stopped plotting at {e}')
            break
    l_=ax.plot(x_array, gauss_peak_sum+yshift, color=fitcolor,label=label)
    plots.append(l_)
    if label:ax.legend()
    return plots



# %%
if __name__=='__main__':
    pass
    