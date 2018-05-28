# wine-classifier-DNN-Keras
These data are the results of a chemical analysis of
wines grown in the same region in Italy but derived from three
different cultivars.
The analysis determined the quantities of 13 constituents
found in each of the three types of wines.
#training parameters
1) Alcohol
2) Malic acid
3) Ash
4) Alcalinity of ash  
5) Magnesium
6) Total phenols
7) Flavanoids
8) Nonflavanoid phenols
9) Proanthocyanins
10)Color intensity
11)Hue
12)OD280/OD315 of diluted wines
13)Proline

I used different config in DNN by adding and removing dropouts,increasing and decreasing learning rate,changing activation function and changing optimizers.
Accuries differs for one another.
You can also change and try it in your own.
Iam getting maximum accuracy at no dropouts,sigmoid activation,adam optimizer with learning rate 0.001.
