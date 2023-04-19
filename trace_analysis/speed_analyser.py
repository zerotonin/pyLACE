import pandas as pd
import numpy as np
import index_tools

class speed_analyser():

    def __init__(self,fps,dataframe = []) -> None:
        self.fps = fps
        self.trace_df = dataframe
        self.allSpeed = []
        self.speed_analysis_df = []
        self.activity = []
        self.cruiseSpeed = []
    
    def get_fish_speeds_from_full_trace(self):
        self.allSpeed = self.trace_df[['thrust_m/s','slip_m/s','yaw_deg/s']]

    def set_activity_array(self,activity_tresh= (.025,.025,100)):
        self.activity = pd.DataFrame([self.allSpeed['thrust_m/s'].abs() > activity_tresh[0],
                                      self.allSpeed['slip_m/s'].abs()   > activity_tresh[1], 
                                      self.allSpeed['yaw_deg/s'].abs()  > activity_tresh[2]]).transpose().any(axis='columns')
    def get_cruise_speed(self):
        self.cruiseSpeed = self.allSpeed[self.activity]

    def calc_torque(self,mode='cruise'):
        if mode == 'cruise':
            torque = np.median((self.cruiseSpeed['thrust_m/s'].abs() + 
                                self.cruiseSpeed['slip_m/s'].abs())  / 
                                self.cruiseSpeed['yaw_deg/s'].abs())
        elif mode == 'all':
            torque = np.median((self.allSpeed['thrust_m/s'].abs() + 
                                self.allSpeed['slip_m/s'].abs())  / 
                                self.allSpeed['yaw_deg/s'].abs())
        else:
            raise ValueError(f'calc_torque:mode unknown: {mode}')
        
        return torque

    def calc_central_speed_values(self,speed_df):
        data = speed_df.abs().mean().tolist()
        data +=  speed_df.abs().median().tolist()
        return data
    
    


    def analyse_fish_speed_df(self):
        
        #rearrange data
        self.get_fish_speeds_from_full_trace()
        self.set_activity_array()
        self.get_cruise_speed()

        #start calculating meta speed values
        data = self.calc_central_speed_values(self.allSpeed)
        data += self.calc_central_speed_values(self.cruiseSpeed)

        act_start_end = index_tools.bool_Seq2start_end_indices(self.activity)
        data.append(self.activity.sum()/self.fps)
        data.append(self.activity.sum()/self.activity.shape[0])
        data.append(self.activity[::-1].idxmax()/self.fps)
        data.append(self.calc_torque())

        keys = ['thrust_mean_m/s', 'slip_mean_m/s', 'yaw_mean_m/s', 'thrust_median_m/s', 'slip_median_m/s', 'yaw_median_m/s', 'cruising_thrust_mean_m/s', 'cruising_slip_mean_m/s', 
                'cruising_yaw_mean_m/s', 'cruising_thrust_median_m/s', 'cruising_slip_median_m/s', 'cruising_yaw_median_m/s', 'activity_duration_s', 'activity_fraction', 'sec_to_first_stop','torque']
        return dict(zip(keys,data))