import pandas as pd

def to_csv(seed_locations,orientations,fname):

    dataframe=pd.DataFrame(seed_locations,columns=['x','y','z'])

    dataframe['q0'],dataframe['q1'],dataframe['q2'],dataframe['q3'] = orientations[:,0],orientations[:,1],orientations[:,2],orientations[:,3]

    dataframe.to_csv(fname,index=False)

    return None