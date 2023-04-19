import data_base_analyser.CounterCurrentAnalyser as CounterCurrentAnalyser
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('/home/bgeurten/fishDataBase/fishDataBase.csv')
cca = CounterCurrentAnalyser.CounterCurrentAnalyser(df)
cca.main('/home/bgeurten/PyProjects/reRandomStats/figures/','/home/bgeurten/PyProjects/reRandomStats/Data/')
plt.show()