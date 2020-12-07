#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))


columns = ["farrah", "Fred", "Felicia"]


plt.bar(columns,fruit[0],0.5,color='red',label="apples")
plt.bar(columns, fruit[1],0.5, color='yellow',bottom=fruit[0],label="bananas")
plt.bar(columns,fruit[2],0.5,color='#ff8000',bottom=fruit[1]+fruit[0],label="oranges")
plt.bar(columns,fruit[3],0.5,color='#ffe5b4',bottom=fruit[0]+fruit[1]+ fruit[2],label="peaches")
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.ylim([0, 80])

plt.show()