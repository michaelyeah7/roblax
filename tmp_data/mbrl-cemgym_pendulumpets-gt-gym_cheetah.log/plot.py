import pickle
import matplotlib.pyplot as plt
import math
import numpy as np
import csv

with open('pendulum_gps.csv', newline='') as f:
    reader = csv.reader(f)
    data = [row[2] for row in reader]
#delete first row of names
data = data[1:]
#conver to list
data = [float(item) for item in data if float(item) > -200 ]
x = np.arange(1,len(data)+1) * 201
plt.plot(x, data)
plt.ylabel("Reward")
plt.xlabel("Episode")
plt.title("PETS_Pendulum")
plt.savefig('pets_pendulum_episode.png')
plt.close()
print(data)
