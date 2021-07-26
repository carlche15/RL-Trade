import numpy as np

class trade_env():
    def __init__(self, data,init_inventory = 10.0, time_range = 10, look_back = 20):
        self.env_data = data
        self.inventory = init_inventory # inventory amount
        self.time_remain = time_range # todo: time unit
        self.look_back = look_back # look back period, which depends on the model
        self.current_time = look_back # the index for the latest data on the sample dataset
        self.current_price = None
        self.oversale_punish = 0 #1e-6 # punish large order
        self.final_liquid = None # the number of shares sold at the last second (forced)
        self.average_trade_price = 0 # average proceed (model VWAP)
        self.TWAP = 0 # market twap


        ### normalization for features # todo: hard code :(!
        self.inv_normalize = 10
        self.t_normalize  = 10
        self.p_normalize = 0.002
        self.v_normalize = 1
        self.normalize = np.array([self.p_normalize,self.v_normalize])[None,None,:]
        
        
        self.action_dir = {
            0: 0.0,
            1: 1.0,
            2: 2.0,
            3: 3.0,
            4: 4.0,
            
        }
        
    
    def start(self):
        """
        return the beginin of state: 1. time_series state; 2. current state
    
        """
        # the original shape is num_of_security(feature) * num_of_price_attr(e.g. close, volume,etc) *time_steps
        # reshape the original data into shape of  num_of_attr*time_steps* num_of_securities
        time_series_input = self.env_data[:,:,self.current_time-self.look_back: self.current_time]
        time_series_input = np.moveaxis(time_series_input, [1,0], [0,2])  
        # current input consisits of current_price, inventory, and time left
        self.current_price = time_series_input[:,-1,0].ravel()[0] # todo: the 1st is the security that we want to trade
        
      

        current_input = np.array([(self.current_price-1)/self.p_normalize, self.inventory/self.inv_normalize,self.time_remain/self.t_normalize])[None,:]  
        self.TWAP = np.sum((self.env_data[0,:,self.current_time-1:self.current_time-1+self.time_remain].ravel())-1)

        
        
        return [(time_series_input-1)/self.normalize,current_input]
    
    def step(self, action_choosen):
        # reward based on previous observed price and action
        
        action = self.action_dir[int(action_choosen)]
        reward = (self.current_price-1)*np.minimum(action, self.inventory) - self.oversale_punish*(action_choosen)**2
        end_of_epsoide = False
        
        ###### save examine data##############
        self.average_trade_price += (self.current_price-1.)*np.minimum(action, self.inventory)
        #######################################
        
  
        
        
        self.current_time = self.current_time+1
        self.inventory = np.maximum(self.inventory - action, 0)
        self.time_remain = self.time_remain -1
        
        
        time_series_input = self.env_data[:,:,self.current_time-self.look_back: self.current_time]
        time_series_input = np.moveaxis(time_series_input, [1,0], [0,2])  
        self.current_price = time_series_input[:,-1,0].ravel()[0] # todo: the 1st is the security that we want to trade
        current_input = np.array([(self.current_price-1)/self.p_normalize, self.inventory/self.inv_normalize,self.time_remain/self.t_normalize])[None,:]
        
        
     
        
        
        
        if self.time_remain == 1 or self.inventory==0: # need to force liquid the asset in the last period or the trade
            # has successfully executed
            liquid_reward = self.inventory*(self.current_price-1.) - self.oversale_punish*(self.inventory)**2
            reward+= liquid_reward
            end_of_epsoide = True
            self.final_liquid = (self.inventory,self.inventory*(self.current_price-1.),self.inventory*self.oversale_punish )
            ###### save examine data##############
            self.average_trade_price += (self.current_price-1.)*self.inventory
            
            #######################################
        
            
            return [np.nan*np.ones_like(time_series_input), np.nan*np.ones_like(current_input)], reward, end_of_epsoide
        else:
            end_of_epsoide = False
            return [(time_series_input-1)/self.normalize,current_input], reward, end_of_epsoide
            
        
        
 