import os
import pandas as pd
from .utils import *
#*Data import functions

@log_function_call
def import_rawdata(file_path: str) -> Optional[pd.DataFrame]:
    '''
    Imports raw data with file types: .chi, .txt.
    Adjust/Add own file type here.
    
    Parameters:
    - file_path: str, path to the file
    
    Returns:
    - pd.DataFrame: the imported data or None if the file doesn't exist or is of an unknown type

    Raises:
    - FileNotFoundError: if the file does not exist
    - ValueError: if the file type is unknown
    '''
    logging.info(f"Attempting to import raw data from {file_path}")
    if os.path.exists(file_path)==True:
        if file_path.endswith('.chi'):
            df=pd.read_table(file_path,skiprows=4, names=["q","I"], sep="\s+")
        elif file_path.endswith('.txt'):
            df=pd.read_table(file_path,skiprows=1, names=["q","I"], sep="\s+")
            
        else:raise ValueError('File type unknown, please specify the type in function import_rawdata()')
        return df    
    
    else:raise FileNotFoundError(f'Could not find file: {file_path}')
@log_function_call





@log_function_call
def load_infos(
    file_path: str,
    sampleinfo: str = 'sampleinfo.xlsx',
    suspensioninfo: str = 'suspensioninfo.xlsx'
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Imports information about scattering files.

    Parameters:
    - file_path : str
        Path to the file
    - sampleinfo : str, optional
        Filename containing sample information.
        Mandatory columns: sample, particlebatch, filedir, filename
    - suspensioninfo : str, optional
        Filename containing suspension information.
        Mandatory columns: particlebatch, filename, dir

    Returns:
    - tuple of dict
        Sample and suspension information dictionaries

    Raises:
    - FileNotFoundError : if the file does not exist
    - ValueError : if the required columns are missing
    """
    def load_suspensions(suspinfo,file_path):
        for key, info in suspinfo.items():
            logging.info(f'try loading diffractogram of suspension {key}')
            f=os.path.join(file_path,info['dir'],info['filename'])
            df=import_rawdata(f)
            suspinfo[key]['data']=df
            
    sampleinfo_path = os.path.join(file_path, sampleinfo)
    if os.path.exists(sampleinfo_path):
        logging.info(f"Trying to load file {sampleinfo_path}")
        sampleinfo_df = pd.read_excel(sampleinfo_path, skiprows=1, dtype={'run': str, 'no': str, "particlebatch": str})
        check_columns(sampleinfo_df, 'sample, particlebatch, dir, filename'.split(', '), sampleinfo)
        sampleinfo=sampleinfo_df.to_dict('index')
        logging.info(f"Sample info file loaded from: {file_path}")
    else:
        raise FileNotFoundError(f"Could not find file: {sampleinfo_path}")

    suspensioninfo_path = os.path.join(file_path, suspensioninfo)
    if os.path.exists(suspensioninfo_path):
        logging.info(f"Trying to load file {suspensioninfo_path}")
        suspinfo_df = pd.read_excel(suspensioninfo_path, dtype={'particlebatch': str})
        check_columns(suspinfo_df, 'particlebatch, filename, dir'.split(', '), suspensioninfo)
        logging.info(f"Suspension info file loaded from: {file_path}")
        suspinfo=suspinfo_df.set_index('particlebatch').to_dict('index')
        load_suspensions(suspinfo,file_path)
    else:
        raise FileNotFoundError(f"Could not find file: {suspensioninfo_path}")
    
    return sampleinfo, suspinfo
        



def get_suspensiondata(sampleid: int, sampleinfo: Dict[int, Any], suspinfo: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Uses the sampleid to find the suspension file of the corresponding batch.

    Parameters:
    - sampleid : int
        The sample identifier.
    - sampleinfo : dict
        Information about the sample.
    - suspinfo : dict
        Information about the suspension.

    Returns:
    - pd.DataFrame or None
        The suspension data for the given sample id.
    """
    if sampleid not in sampleinfo:
        logging.warning(f"Sample ID {sampleid} not found in sample info.")
        return None

    batch = sampleinfo[sampleid]['particlebatch']

    if batch not in suspinfo:
        logging.warning(f"Batch {batch} not found in suspension info.")
        return None

    return suspinfo[batch]['data']