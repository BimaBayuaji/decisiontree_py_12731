import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

# import dataset
iris = datasets.load_iris()
features = iris['data']
target = iris['target']
print(target)

# objek model
decisiontree = DecisionTreeClassifier(random_state=0, 
        max_depth=None, min_samples_split=2, min_samples_leaf=1,
        min_weight_fraction_leaf=0, max_leaf_nodes=None, min_impurity_decrease=0)

# training model
model = decisiontree.fit(features, target)

# predic
observation = [[5, 4, 3, 2]]
model.predict(observation)
model.predict_proba(observation)

# visualisasi model
import pydotplus
from sklearn import tree
dot_data = tree.export_graphviz(decisiontree, out_file=None,
                                feature_names=iris['feature_names'], class_names=iris['target_names'])
from IPython.display import Image
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_png('iris.png')

# Bima Bayuaji - A11.2020.12731