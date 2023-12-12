"""
    File to load dataset based on user control from main file
"""
from data.molecules import MoleculeDatasetDGL

def LoadData(DATASET_NAME, parmas):
    """
        This function is called in the main_xx.py file 
        returns:
        ; dataset object
    """    
    # handling for (ZINC) molecule dataset
    if DATASET_NAME == 'CO' or DATASET_NAME == 'H' or DATASET_NAME == 'N' or DATASET_NAME == 'O' or DATASET_NAME == 'OH' or DATASET_NAME == 'OOH':
        return MoleculeDatasetDGL(DATASET_NAME, parmas['seed'], parmas['pklname'])
    else:
        return None