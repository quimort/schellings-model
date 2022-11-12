import numpy as np
import csv
supreme_array = np.array([])
with open("schelling_values_100.csv","r") as file:
    csv_reader = csv.reader(file, delimiter=';')
    line_count = 0
    for row in csv_reader:
        if(line_count !=0):
            if(len(row)!=0):
                construct_array = np.array([float(row[0]),float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6]),float(row[7])])
                supreme_array = np.append(supreme_array,construct_array)
        line_count +=1
supreme_array = supreme_array.reshape((int(np.size(supreme_array)/8),8))
vacant = np.unique(supreme_array[:,0])
f = open("schelling_values_100__50_mean.csv", "w")
f.write("vacant;similarity ratio inicial;mean dissatisfaction inicial;mean interratial pears inicial\
    ;similarity ratio final;mean dissatisfaction final;mean interratial pears final;number of iterations")
f.write("\n")
f.close
for empty in vacant:
    count = 0
    values = np.zeros(8)
    for i in range(supreme_array.shape[0]):
        if(supreme_array[i][0] == empty):
            for ii in range(0,7):
                values[ii] = values[ii] + supreme_array[i][ii+1]
            count +=1
    values_final = values/count
    f = open("schelling_values_100__50_mean.csv", "a")
    f.write("{};{};{};{};{};{};{};{}".format(empty,values_final[0],values_final[1],values_final[2],values_final[3],values_final[4],values_final[5],values_final[6]))
    f.write("\n")
    f.close
print("a")
