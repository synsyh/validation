import xgboost as xgb
# read in data
dtrain = xgb.DMatrix('./data/test')
dtest = xgb.DMatrix('./data/train')
# specify parameters via map
param = {'max_depth': 6, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 2
bst = xgb.train(param, dtrain, num_round, watchlist)
