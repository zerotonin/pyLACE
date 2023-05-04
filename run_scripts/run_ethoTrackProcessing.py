from data_base_analyser.EthoVisionExperimentSeries import EthoVisionExperimentSeries
import os
import numpy as np




# Usage
tag = 'habituation2023'
parent_directory = '/home/bgeurten/ethoVision_database/'
etho_vision_analysis = EthoVisionExperimentSeries(tag, parent_directory)
etho_vision_analysis.process_and_save()


