import numpy as np
from genetic_decision_tree import GeneticDecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def get_data(path: str):
    data = []
    with open(path) as file:
        n = int(file.readline())
        line = file.readline()
        while line != '':
            d = line.split(' ')
            assert len(d) == n + 1, f'each line of file must contain {n + 1} integers'
            data.append([int(d[i]) for i in range(n + 1)])
            line = file.readline()

    return np.array(data)


if __name__ == '__main__':
    data = get_data('inputs/test05.txt')

    X = data[:, :-1]
    y = data[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    gdtc = GeneticDecisionTreeClassifier(max_depth=4,
                                         random_state=5,
                                         genetic_max_iterations=10,
                                         genetic_k=8)
    gdtc.fit(X_train, y_train)
    y_pred = gdtc.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}% \n')
    print(f'Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))

