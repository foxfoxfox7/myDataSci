# THE DESNITY OF A SCATTER PLOT ###############################################

from scipy.stats import gaussian_kde

x_yes = df.Age.values
y_yes = df.Yes.values

# Calculate the point density
xy = np.vstack([x_yes,y_yes])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
ax.scatter(x_yes, y_yes, c=z, s=100, edgecolor='')
plt.show()

# ADDING JITTER TO PLOTS ######################################################
# OR MAKING IT SO ITS NOT ALL THE SAME VALUE

#y_yes = y_yes+0.00001*np.random.rand(len(y_yes))

