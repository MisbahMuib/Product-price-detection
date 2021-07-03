import matplotlib.pyplot as plt

sizes = 54 ,71 ,53, 57,68
explode = (0.013, 0.013,0.013, 0.013,0.013, )
labels = [ 'harpic', 'lux','Toothpowder','pepsodent','vim']
#autopact show percentage inside graph
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',)
plt.axis('equal')
plt.savefig('percentage of products .jpg') # need to call before calling show
plt.show()
