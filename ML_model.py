from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LR
from xgboost import XGBClassifier
from sklearn import tree, svm
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB


def rf_classifier(X_train, Y_train, seed):
    rfmodel = RandomForestClassifier(
                                     n_estimators=40,
                                     max_depth=6,
                                     min_samples_split=10,
                                     min_samples_leaf=12,
                                     min_weight_fraction_leaf=0.04612871218113838,
                                     oob_score=True,
                                     class_weight={1:2, 0: 1},
                                     n_jobs=-1,
                                     random_state=seed
                                     )

    rfmodel.fit(X_train,
                Y_train)
    # y_pred = rfmodel.predict(X_valid)
    # y_score = rfmodel.predict_proba(X_valid)[:, 1]  ##预测概率

    return rfmodel


def lr_classifier(X_train, Y_train, seed):
    lr = LR(solver='liblinear',
            max_iter=100,
            C=0.2,
            class_weight={0: 1, 1: 2},
            random_state=seed)
    lr.fit(X_train, Y_train)
    return lr


def xgb_classifier(X_train, Y_train, seed):
    xgbmodel = XGBClassifier(
        objective="binary:logistic",
        learning_rate=0.17833175686361005,
        n_estimators=116,
        subsample=0.5989596612268187,
        max_depth=7,
        min_child_samples=22,
        min_child_weight=0.031908434686626595,
        reg_alpha=4.168860771947327,
        reg_lambda=0.7655954488471695,
        random_state=seed,
        class_weight={1: 35, 0: 1},
        # scale_pos_weight=38,
        colsample_bytree=0.7506118870766936
    )
    xgbmodel.fit(X_train,
                 Y_train,
                 # eval_set=[(X_valid, Y_valid)],
                 # early_stopping_rounds=50,
                 verbose=False
                 )
    return xgbmodel


def lgb_classifier(X_train, Y_train, seed):
    gbm = LGBMClassifier(boosting_type='gbdt',
                         objective='binary',
                         # eval_metric='Recall',
                         max_depth=14,
                         min_child_samples=12,
                         min_child_weight=0.03436315913966144,
                         learning_rate=0.06442395847713227,
                         num_leaves=12,
                         feature_fraction=0.594971207159772,
                         n_estimators=52,
                         bagging_fraction=0.6499665482435462,
                         # is_unbalance=False,
                         class_weight={1: 35, 0: 1},
                         # scale_pos_weight=37,
                         lambda_l1=1.1414550610890912,
                         lambda_l2=2.9569179269872787,
                         n_jobs=-1,
                         random_state=seed
                         )

    gbm.fit(X_train,
            Y_train,
            # eval_set=[(X_valid, Y_valid)],
            # eval_metric='Recall',
            # early_stopping_rounds=50,
            verbose=False)
    return gbm


def ctb_classifier(X_train, Y_train, seed):
    ctb = CatBoostClassifier(
        iterations=430,
        learning_rate=0.07089446779738615,
        depth=12,
        loss_function='Logloss',
        # use_best_model=True,
        custom_metric=['Recall', 'AUC', 'Precision', 'Accuracy', 'ZeroOneLoss'],
        # eval_metric='Recall',
        min_child_samples=8,
        reg_lambda=1.814831726181313,
        subsample=0.8576119241716049,
        rsm=0.5726677559696667,
        random_seed=seed,
        class_weights={0: 1, 1: 9},
        # early_stopping_rounds=100,
        verbose=False
    )
    ctb.fit(X_train,
            Y_train,
            # eval_set=[(X_valid, Y_valid)],
            # early_stopping_rounds=100,
            verbose=False)
    return ctb


def dt_classifier(X_train, Y_train, seed):
    dt = tree.DecisionTreeClassifier(criterion="gini",
                                     random_state=seed)
    dt.fit(X_train, Y_train)
    return dt


def svm_classifier(X_train, Y_train, seed):
    svc = svm.SVC(random_state=seed)
    svc.fit(X_train, Y_train)
    return svc


def knn_classifier(X_train, Y_train, seed):
    knn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
    knn.fit(X_train, Y_train)
    return knn


def gnb_classifier(X_train, Y_train, seed):
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
    return gnb
