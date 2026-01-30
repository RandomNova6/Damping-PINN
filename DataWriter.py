import pandas as pd

class DataWriter():
    def __init__(self,filepath):
        self.filepath=filepath  

    def save_data(self,df):
        df.columns=['theta','T','phi']
        df.to_excel(self.filepath, index=False)

if __name__ == "__main__":
    '''theta=[45.5, 127.4, 49.5, 122.6, 54.0, 112.6, 70.6, 93.2, 91.0, 92.3, 93.0, 87.6, 95.9, 63.0, 108.5, 53.0, 120.0, 49.0, 127.3, 44.5, 130.5, 42.5]
    T=[1.8697, 1.7597, 1.8597, 1.7554, 1.85, 1.7503, 1.8246, 1.743, 1.8042, 1.7422, 1.8029, 1.7403, 1.8002, 1.7249, 1.7905, 1.715, 1.7809, 1.7103, 1.7749, 1.7021, 1.7699, 1.6998]
    phi= [26.0, 110.0, 27.0, 119.0, 30.0, 128.0, 38.0, 139.0, 49.0, 141.0, 51.0, 143.0, 53.0, 157.0, 61.0, 162.0, 72.0, 163.0, 81.0, 165.0, 89.0, 166.0]
    data=zip(theta,T,phi)
    listed_data=list(data)
    listed_data.sort(key=lambda row: row[1])
    data0 = pd.DataFrame(listed_data,columns=['theta','T','phi'])

    expData=pd.read_excel('Experiment.xlsx',header=None)
    data17=expData.iloc[:,1:4].dropna()
    num_rows_to_keep = len(data17) - 14
    data17_reduced = data17.sample(n=num_rows_to_keep, random_state=42)    
    data18=expData.iloc[:,5:8].dropna()
    data14=expData.iloc[:,9:12].dropna()
    data15=expData.iloc[:,13:16].dropna()
    data16=expData.iloc[:,17:20].dropna()

    data17_reduced.columns=['theta','T','phi']
    data17.columns=['theta','T','phi']
    data18.columns=['theta','T','phi']
    data14.columns=['theta','T','phi']
    data15.columns=['theta','T','phi']
    data16.columns=['theta','T','phi']

    data17.sort_values(by='T',inplace=True)
    data18.sort_values(by='T',inplace=True)
    data14.sort_values(by='T',inplace=True)
    data15.sort_values(by='T',inplace=True)
    data16.sort_values(by='T',inplace=True)

    data_frames=[data0,data14,data15,data16,data17]
    combined_data = pd.concat(data_frames, ignore_index=True)
    print(combined_data)'''

    expData17=pd.read_excel('Experiment17.xlsx',header=None)
    data17_1=expData17.iloc[:,0:3]
    data17_2=expData17.iloc[:,3:6]
    data17_3=expData17.iloc[:,6:9]

    writer=DataWriter('ExperimentData.xlsx')
    writer.save_data(data17_3)