import numpy as np

def forward(x, w, b) -> np.array:
    x = np.array(x)
    w = np.array(w)
    b = np.array(b)
    
    xw_dot = np.dot(x, w)
    z = xw_dot + b
    
    return z

def relu(z : np.ndarray) -> np.ndarray:
    return np.maximum(0,z)

def layer_forward(x, w, b):
    z = forward(x, w, b)
    return relu(z)

if __name__ == "__main__":
    x = np.array([1.0, 2.0, -1.0])
    
    w = np.array([
        [0.5, -0.1],
        [0.2, 0.8],
        [-0.5, 0.3]
    ])
    
    b = np.array([0.1, -0.2])
    
    output = layer_forward(x, w, b)
    print("Activated Output:", output)