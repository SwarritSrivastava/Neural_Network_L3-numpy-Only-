import json
import numpy as np

def initialize(filename):
    def label_encoder(y):
        y_encoded = []
        for gesture in y:
            if(gesture == "thumbs_up"):
                y_encoded.append(2)
            elif(gesture == "fcuk_off"):
                y_encoded.append(3)
            elif(gesture == "open"):
                y_encoded.append(0)
            elif(gesture == "close"):
                y_encoded.append(1)
            
        return y_encoded

    with open(filename, "r") as f:
        data = json.load(f)
    for obesrvation in data:    
        palmpoint = obesrvation["landmarks"][0]
        x0 = palmpoint[0]
        y0 = palmpoint[1]
        z0 = palmpoint[2]

        anchor = [x0 , y0, z0]

        for coordinates in obesrvation["landmarks"]:
            for i in range (3):
                coordinates[i] = coordinates[i] - anchor[i]

    X = [np.array(obs["landmarks"]).flatten() for obs in data]
    y = [obs["gesture"] for obs in data]
    y_encoded = label_encoder(y)
    X = np.array(X)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-7)
    
    return X,y_encoded