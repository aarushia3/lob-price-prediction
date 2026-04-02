import numpy as np

def process_labels(y):
    return (y - 1).astype(int)  # Convert from 1,2,3 to 0,1,2