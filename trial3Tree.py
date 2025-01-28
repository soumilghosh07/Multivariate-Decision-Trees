import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression

class Node:
    def __init__(self, feature1=None, feature2=None, acoeff=None, bcoeff=None, ccoeff=None, threshold=None, left=None, right=None, *, value=None):
        self.feature1 = feature1
        self.feature2 = feature2
        self.acoeff = acoeff
        self.bcoeff = bcoeff
        self.ccoeff = ccoeff
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

        
    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # find the best split
        trans_col, best_thresh, feat1, feat2, a2,b2,c2 = self._best_split(X, y, feat_idxs)

        # create child nodes
        left_idxs, right_idxs = self._split(trans_col, best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(feat1,feat2, a2,b2,c2,best_thresh, left, right)
    
    
    def _linear_combination(self, X,y, feat_idxs):
        # Extract the best feature indices
        # Fit logistic regression to learn coefficients
        feature1_idx, feature2_idx = np.random.choice(feat_idxs, size=2, replace=False)
        lr = LogisticRegression(max_iter=10000)
        lr.fit(X[:, (feature1_idx, feature2_idx)], y)
        a = lr.coef_[0][0]
        b = lr.coef_[0][1]
        c = lr.intercept_[0]
        
        
        # Perform linear split using the selected features
        return a * X[:, feature1_idx] + b*X[:, feature2_idx] + c,feature1_idx,feature2_idx,a,b,c  # Example linear equation: 2X1 - X2 - 5 = 0



    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        X_linear_output,chosenfeature1,chosenfeature2,a,b,c = self._linear_combination(X,y, feat_idxs)
        thresholds = np.unique(X_linear_output)
        for thr in thresholds:
            # calculate the information gain
            gain = self._information_gain(y, X_linear_output, thr)
            if gain > best_gain:
                best_gain = gain
                split_idx = X_linear_output
                split_threshold = thr
                cf1=chosenfeature1
                cf2=chosenfeature2
                a1=a
                b1=b
                c1=c
        return split_idx, split_threshold,cf1,cf2,a1,b1,c1


    def _information_gain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # calculate the weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])


    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if ( node.acoeff * x[node.feature1] + node.bcoeff * x[node.feature2] + node.ccoeff) <= node.threshold:          # return 2 * X[:, feature1_idx] - X[:, feature2_idx] - 5,feature1_idx,feature2_idx  # Example linear equation: 2X1 - X2 - 5 = 0
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
        


