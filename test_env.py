import gym, safety_gym

# from safety_gym.envs.engine import Engine

# config = {
#     'robot_base': 'xmls/car.xml',
#     'task': 'button',   
# }

# env = Engine(config)

# from gym.envs.registration import register

# register(id='SafexpTestEnvironment-v0',
#          entry_point='safety_gym.envs.mujoco:Engine',
#          kwargs={'config': config})

env = gym.make('Safexp-PointGoal1-v0')
o= env.reset()
print(env.action_space)
for i in range(10000):
    env.render()
    a = env.action_space.sample()
    # env.action_space()
    o2, r, d, info = env.step(a)
    print(o2)
    
    if d:
        env.reset()
        break
    # env.reset()
    print(f"{i}\t{info}")

env.reset()