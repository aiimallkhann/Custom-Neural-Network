import numpy as np

def forward(x, w, b) -> np.array:
    x = np.array(x)
    w = np.array(w)
    b = np.array(b)
    
    xw_dot = np.dot(x, w)
    z = xw_dot + b
    
    return z