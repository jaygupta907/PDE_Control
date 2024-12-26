import json
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from jaxfluids import InputReader, Initializer, SimulationManager
import stable_baselines3 as sb3
import gymnasium as gym
from gymnasium import spaces
import csv

class NavierStokes(gym.Env):
    def __init__(self,reference_state,steps):
        super(NavierStokes,self).__init__()
        self.case_setup        = json.load(open("flow.json"))
        self.numerical_setup   = json.load(open("flow_numerical_setup.json"))
        self.input_reader      = InputReader(self.case_setup,self.numerical_setup)
        self.reference_state   = reference_state
        self.steps             = steps
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(64,64), dtype=np.float32)
        self.t         = 0
        self.rewards = []
        self.timesteps = []
        self.actions =   []

    
    def step(self,action):
        self.sim_manager.boundary_condition.dirichlet_functions['west']['u'] = action.item()
        for i in range(self.steps):
            current_time      = self.buffer_dictionary['time_control']['current_time']
            timestep          = self.sim_manager.compute_timestep(
                                        **self.buffer_dictionary["material_fields"],
                                        **self.buffer_dictionary["levelset_quantities"]
                                        )
            material_field,level_set,residuals   = self.sim_manager.do_integration_step(
                                                **self.buffer_dictionary["material_fields"],
                                                **self.buffer_dictionary["levelset_quantities"],
                                                timestep_size=timestep,
                                                current_time=current_time)
            self.buffer_dictionary["material_fields"].update(material_field)
            self.buffer_dictionary["levelset_quantities"].update(level_set)
            self.buffer_dictionary["time_control"]["current_time"] = current_time + timestep
            
        next_state = self.buffer_dictionary['material_fields']['primes'][1][5:-5,5:-5,0]
        self.reward =  self.get_reward(next_state,self.reference_state,action)
        self.done  =  self.is_done(next_state,self.reference_state)
        self.t+=1
        print(f'Current Time is {self.t}')
        print(f'Action is {action.item()}')
        print(f'Reward is {self.reward}')
        print(f'Current State is: ')
        print(next_state)
        print('<======================================================================>')
        self.rewards.append(self.reward)
        self.timesteps.append(self.t)
        self.actions.append(action.item())
        return next_state,self.reward,self.done,False,{}
    

    def reset(self,seed=0.5):
        self.initializer       = Initializer(self.input_reader)
        self.buffer_dictionary = self.initializer.initialization()
        self.sim_manager       = SimulationManager(self.input_reader)
        self.sim_manager.initialize(self.buffer_dictionary)
        self.t = 0
        self.sim_manager.boundary_condition.dirichlet_functions['west']['u'] = seed
        self.timesteps = []
        self.rewards = []
        self.actions = []
        return self.buffer_dictionary['material_fields']['primes'][1][5:-5,5:-5,0],{}
    
    def get_reward(self,next_state,reference_state,action):
        reward = -0.1*np.sum((next_state-reference_state)**2) - np.sum(action**2)
        return reward

    def render(self,type):
        # self.volume_fraction = self.buffer_dictionary['levelset_quantities']['volume_fraction']
        # self.levelset = self.buffer_dictionary['levelset_quantities']['levelset']
        # self.mask = self.sim_manager.levelset_handler.compute_masks(levelset=self.levelset,volume_fraction=self.volume_fraction)
        # self.mask  =  self.mask[0][3:-3,3:-3,0]

        if type =='train':
            csv_filename = "ppo_data_1000.csv"
            with open(csv_filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["rewards", "action"])
                for reward, action in zip(self.rewards, self.actions):
                    writer.writerow([reward, action])
    
        plt.figure()
        plt.plot(self.timesteps,self.rewards,c='r')
        plt.xlabel('Timesteps')
        plt.ylabel('Rewards')
        plt.grid()
        plt.show()

        plt.figure()
        plt.plot(self.timesteps,self.actions,c='r')
        plt.xlabel('Timesteps')
        plt.ylabel('Actions')
        plt.grid()
        plt.show()

        plt.figure()
        plt.imshow(self.buffer_dictionary['material_fields']['primes'][1][5:-5,5:-5,0])
        plt.colorbar()
        plt.show()



    def is_done(self,next_state,reference_state):
        return np.sum((next_state-reference_state)**2) < 1e-2