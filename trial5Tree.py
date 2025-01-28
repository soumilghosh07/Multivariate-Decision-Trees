import numpy as np
from collections import Counter
from sklearn.feature_selection import SelectKBest, f_regression
    
from sklearn.linear_model import LogisticRegression, Lasso

class Node:
    def __init__(self, feature=None, coeff=None, intercept=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.coeff = coeff
        self.intercept = intercept
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

        

        selector = SelectKBest(score_func=f_regression, k=self.n_features)
        X_selected = selector.fit_transform(X, y)
        feat_idxs = selector.get_support(indices=True)
        

        # find the best split
        trans_col, best_thresh, feat,coefficients,intercept = self._best_split(X_selected, y, feat_idxs)

        # create child nodes
        left_idxs, right_idxs = self._split(trans_col, best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(feat,coefficients,intercept,best_thresh, left, right)
    


    def _linear_combination(self, X, y, feat_idxs):
        # Fit logistic regression with Lasso regularization to learn coefficients and select features
        lasso = Lasso(alpha=0.1, max_iter=10000)  # You can adjust the alpha value as needed
        lasso.fit(X[:, feat_idxs], y)
        
        # Extract the indices of non-zero coefficient features selected by Lasso
        selected_feat_idxs = np.where(lasso.coef_ != 0)[0]
        
        # Fit logistic regression with selected features
        lr = LogisticRegression(max_iter=10000)
        lr.fit(X[:, selected_feat_idxs], y)
        
        # Extract coefficients and intercept
        coefficients = lr.coef_[0]
        intercept = lr.intercept_[0]
        
        # Compute linear combination using the selected features, coefficients, and intercept
        linear_output = np.dot(X[:, selected_feat_idxs], coefficients) + intercept
        
        return linear_output, selected_feat_idxs, coefficients, intercept
    


    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        X_linear_output,chosenfeatures,coeffs,c = self._linear_combination(X,y, feat_idxs)
        thresholds = np.unique(X_linear_output)
        for thr in thresholds:
            # calculate the information gain
            gain = self._information_gain(y, X_linear_output, thr)
            if gain > best_gain:
                best_gain = gain
                split_idx = X_linear_output
                split_threshold = thr
                chos=chosenfeatures
                coe=coeffs
                intc=c
        return split_idx, split_threshold,chos,coe,intc


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
    
        # Use the same subset of features used during tree growth
        selected_feat_idxs = node.feature
        linear_output = np.dot(x[selected_feat_idxs], node.coeff) + node.intercept
        if linear_output <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    

        


