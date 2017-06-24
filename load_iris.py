from sklearn.datasets import load_iris
iris_dataset=load_iris()

print("Schluessel von iris_dataset: \n {}".format(iris_dataset.keys())+"\n\n"),

print(iris_dataset['DESCR'][:193] + "\n..."+"\n\n"),

print("Zielbezeichnungen: {}".format(iris_dataset['target_names'])+"\n\n"),

print("Namen der Merkmale: \n{}".format(iris_dataset['feature_names'])+"\n\n"),

print("Typ der Daten: {}".format(type(iris_dataset['data']))+"\n\n"),

print("Abmessung der Daten: {}".format(iris_dataset['data'].shape)+"\n\n"),

print("Die ersten fuenf Zeilen der Daten:\n{}".format(iris_dataset['data'][:5])+"\n\n"),

print("Typ der Zielgroessee: {}".format(type(iris_dataset['target']))+"\n\n"),

print("Abmessungen der Zielgroesse: {}".format(iris_dataset['target'].shape)+"\n\n"),

print("Zielwerte:\n{}".format(iris_dataset['target'])+"\n\n"),