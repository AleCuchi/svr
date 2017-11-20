import matplotlib.pyplot as plt
import time
import datetime


var = []
for j in range(5):
    i = time.strptime('Jun '+str(j+1)+' 2017  1:33PM', '%b %d %Y %I:%M%p')
    string = str(i.tm_mday) +'/'+str(i.tm_mon)+'/'+str(i.tm_year)
    var.append(str(string))
plt.plot(var, [2,3,4,5,6], 'red',[6,7,8],[1,2,3],'blue')
plt.show()