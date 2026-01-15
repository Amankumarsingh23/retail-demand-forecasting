import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

def train_model(X_train, y_train, X_val, y_val):
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        early_stopping_rounds=50,
        verbose=False
    )

    val_preds = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_preds)

    return model, val_mae
