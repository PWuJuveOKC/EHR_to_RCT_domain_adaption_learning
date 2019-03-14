import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


class QLearn:
      
      def __init__(self, alpha=0, random_state=0, propensity = 0.5, n_estimators=100):

        self.alpha = alpha
        self.propensity = propensity
        self.n_estimators = n_estimators
        self.random_state = random_state


      def fit(self, X, Q, A):

        self.Qreg = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
        self.p = X.shape[1]
        QL_Int = X * np.transpose(np.tile(A, (self.p, 1)))
        self.Qreg.fit(np.concatenate((X, QL_Int, A.reshape(-1, 1)), axis=1), Q)

        return self

      def predict(self,X):

        np.random.seed(self.random_state)
        A1 = np.ones(len(X))
        QL_Int1 = X * np.transpose(np.tile(A1, (self.p, 1)))
        Q1 = self.Qreg.predict(np.concatenate((X, QL_Int1, A1.reshape(-1, 1)), axis=1))
        A2 = - np.ones(len(X))
        QL_Int2 = X * np.transpose(np.tile(A2, (self.p, 1)))
        Q2 = self.Qreg.predict(np.concatenate((X, QL_Int2, A2.reshape(-1, 1)), axis=1))
        classification = np.sign(Q1 - Q2)

        return classification

      def estimate(self, X, Q, A, normalize=True):

          np.random.seed(self.random_state)
          A1 = np.ones(len(X))
          QL_Int1 = X * np.transpose(np.tile(A1, (self.p, 1)))
          Q1 = self.Qreg.predict(np.concatenate((X, QL_Int1, A1.reshape(-1, 1)), axis=1))
          A2 = - np.ones(len(X))
          QL_Int2 = X * np.transpose(np.tile(A2, (self.p, 1)))
          Q2 = self.Qreg.predict(np.concatenate((X, QL_Int2, A2.reshape(-1, 1)), axis=1))
          classification = np.sign(Q1 - Q2)

          if self.propensity == 'obs':
              logist = linear_model.LogisticRegression()
              logist.fit(X, A)
              prob = logist.predict_proba(X)[:, 1]
          else:
              prob = self.propensity
          PS = prob * A + (1 - A) / 2

          Q0 = Q[np.where(A == classification)]
          PS0 = PS[np.where(A == classification)]

          if not normalize:
            est_Q = np.sum(Q0 / PS0) / len(Q)
          elif normalize:
            est_Q = np.sum(Q0 / PS0) / np.sum(1 / PS0)

          return est_Q
