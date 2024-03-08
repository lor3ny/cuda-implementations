import csv
import random



n = 1<<20

random_list = [random.randint(0, 9) for _ in range(n)]
    
with open('profiles20.csv', 'w', newline='') as file:
    writer = csv.writer(file)
          
    writer.writerow(random_list)
