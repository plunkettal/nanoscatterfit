import pandas as pd
import os
from typing import Tuple, Optional,Dict,Any,List
from scipy import integrate
from scipy.signal import find_peaks
import logging
logging.basicConfig(
    level=logging.INFO,
    filename='nanoscatterfit.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
from functools import wraps





#* general functions

def log_function_call(func):
    '''
    automatically logs used functions when @log_function_call is used above a function.
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Log function call with trimmed arguments
        trimmed_args = [type(arg) if len(str(arg))>100 else arg for arg in args ]
        trimmed_kwargs = {k: type(v) if len(str(v))>100 else v for k, v in kwargs.items()}
        logging.info(f"Called function {func.__name__} with args: {trimmed_args} and kwargs: {trimmed_kwargs}")
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logging.exception(f"Function {func.__name__} raised an exception")
            raise
        # Log return value, trimmed if it's a DataFrame
        trimmed_result = type(result) if len(str(result))>100 else result
        logging.info(f"Function {func.__name__} returned {trimmed_result}")
        return result
    return wrapper


def delete_log_file(file_path: str='nanoscatterfit.log') -> None:
    """
    Deletes the specified log file.

    Parameters:
    - file_path : str
        The path to the log file to be deleted.

    Returns:
    - None
    """
    # logging.shutdown()
    try:
        
        with open(file_path, 'w') as f:
            #rewrite the content of the log file with the empty sting
            f.write('')
            f.close()
        print(f"Log file {file_path} has been deleted.")
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except PermissionError:
        print(f"Permission denied. Could not delete file {file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")
def check_columns(df: pd.DataFrame, required_columns: list, file_name: str):
    """Check if required columns exist in the DataFrame."""
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} is missing in the file {file_name}.")


#* data processing


@log_function_call
def cut_diffractogram(df: pd.DataFrame, qmin: float = 0, qmax: float = 1.5) -> Optional[pd.DataFrame]:
    """
    Cuts the diffractogram into the desired q range specified by qmin and qmax.

    Parameters:
    - df : pd.DataFrame
        DataFrame containing the diffractogram data. Must have a 'q' column.
    - qmin : float, optional
        Minimum q value for the diffractogram cut. Default is 0.
    - qmax : float, optional
        Maximum q value for the diffractogram cut. Default is 1.5.

    Returns:
    - pd.DataFrame or None
        DataFrame containing the cut diffractogram, or None if the input DataFrame is empty or invalid.
    """

    if df.empty or 'q' not in df.columns:
        logging.warning("Input DataFrame is empty or does not contain a 'q' column.")
        return None

    cut_df = df.loc[(df['q'] > qmin) & (df['q'] < qmax)].reset_index(drop=True)
    if cut_df.empty:
        logging.warning(f"No data points found in the range qmin={qmin}, qmax={qmax}.")
        return None

    return cut_df

# @log_function_call
def isscatter(df: pd.DataFrame, intensitytheshhold: int = 20) -> Optional[bool]:
    """
    Checks if the file is actually a scattering file by finding whether the first peak is in a reasonable q range.
    Adjust according to data structure.

    Parameters:
    - df : pd.DataFrame
        DataFrame containing the spectrum data. Must have a 'q' and 'I' column.
    - qmin : float, optional
        Minimum q value for checking the peak. Default is 0.
    - qmax : float, optional
        Maximum q value for checking the peak. Default is 0.3.

    Returns:
    - bool or None
        True if the first peak is within the specified q range, False otherwise.
    """

    if df.empty or 'q' not in df.columns or 'I' not in df.columns:
        logging.warning("Input DataFrame is empty or does not contain 'q' and/or 'I' columns.")
        return None

    # logging.info(f"Trying to check if the diffractogram has a peak between {qmin} and {qmax}")
    try:
        id_max = df['I'].idxmax()
        peak_q_value = df.loc[id_max, 'q']
        peaks=find_peaks(df.S, prominence=0.1)
        # if qmin < peak_q_value < qmax:
        logging.info('Checking if intensity at larger q values is at ca. 10')
        meanintensity=df.loc[(df.q > 2* peak_q_value)&(df.q < 5* peak_q_value)].I.mean()
        if meanintensity < intensitytheshhold:
            logging.info(f'The mean intensity of the diffractogram is {meanintensity:.0f}, and presumably not a diffraction file.')
            return False
        elif peaks[0][0] != df.S.idxmax():
            logging.info('Checking if 111 peak is largest')
            
            logging.info(f'The 111 peak was unexpectedly small.')
            return False
        else:
            logging.info(f'The mean intensity of the diffractogram is {meanintensity:.0f}, and presumably a diffraction file.')
            return True
    except Exception as e:
        logging.exception(f"An error occurred: {e}")
        return False
    # else:
    #     logging.info(f"The peak is not in the range between {qmin} and {qmax}. It's at q = {peak_q_value}.")
    #     return False
    

def normalize_area(x, y):
    """
    Normalizes the area under the curve represented by x and y.

    Parameters:
    - x : array-like
        The x values.
    - y : array-like
        The y values.

    Returns:
    - array-like
        The y values normalized such that the area under the curve is 1.
    """
    integral = integrate.trapz(y, x)
    return y / integral



@log_function_call
def auto_cutdiffr(x:pd.Series,y:pd.Series,maxpeakposition:float)->Optional[pd.Series]:
    '''
    cuts the diffractogram at the first and last minimum to improve fitting process.
    Ideally data is normalized to area=1 and smoothed.
    
    Parameters:
    - y : pd.Series
        Series containing the diffractogram data.

    Returns:
    - pd.Series 
        reduced diffractorgram
    '''
    
    y=y.loc[(x < maxpeakposition)]
    x=x.loc[(x < maxpeakposition)]
    logging.info(f'precut from {x.iloc[0]} to {x.iloc[-1]}')
    # logging.info(y.iloc[-5:])
    for i in (y.index):
        # if i==0:continue
        if (y[i] <= y[i+1])&(y[i] < y[i+2]):
            start=i
            break
    for i in reversed(y.index):

        if (y[i] <= y[i-1])&(y[i] < y[i-2]):
            y[i]
            end=i
            break
    logging.info(f'start={x[start]}, end={x[end]}')
    if start==end:
        logging.error(f'couldnt autocut diffractogram')
        return None,None
    return x[start:end],y[start:end]