import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class RetailDemandForecaster:
    def __init__(self, forecast_horizon=4, lookback=8):
        self.forecast_horizon = forecast_horizon
        self.lookback = lookback
        self.scalers = {}
        
    def prepare_data(self, df):
     
        df = df.sort_values(['week', 'store_id', 'sku_id'])
        
       
        df['price_ratio'] = df['total_price'] / df['base_price']
        df['price_diff'] = df['total_price'] - df['base_price']

        df['promo_combo'] = df['is_featured_sku'] + df['is_display_sku']
        groups = df.groupby(['store_id', 'sku_id'])
        
        lag_features = ['units_sold', 'price_ratio', 'price_diff']
        for col in lag_features:
            for lag in range(1, self.lookback + 1):
                df[f'{col}lag{lag}'] = groups[col].shift(lag)
        
        windows = [2, 4, 8]
        for col in lag_features:
            for window in windows:
                df[f'{col}rolling_mean{window}'] = groups[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean())
        
        return df.dropna()

    def create_sequences(self, data, target_col='units_sold'):
        feature_cols = [col for col in data.columns if col not in 
                       ['record_ID', 'week', 'store_id', 'sku_id', 'units_sold']]
        
        X, y = [], []
        groups = data.groupby(['store_id', 'sku_id'])
        
        for _, group in groups:
            for i in range(len(group) - self.forecast_horizon - self.lookback + 1):
                X.append(group[feature_cols].iloc[i:i+self.lookback].values)
                y.append(group[target_col].iloc[i+self.lookback:i+self.lookback+self.forecast_horizon].values)
        
        return np.array(X), np.array(y)

    def build_model(self, input_shape, output_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(output_shape)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def fit(self, file_path):

        df = pd.read_csv(file_path)
        processed_df = self.prepare_data(df)
        feature_cols = [col for col in processed_df.columns if col not in 
                       ['record_ID', 'week', 'store_id', 'sku_id', 'units_sold']]
        
        for col in feature_cols + ['units_sold']:
            self.scalers[col] = StandardScaler()
            processed_df[col] = self.scalers[col].fit_transform(processed_df[[col]])
        X, y = self.create_sequences(processed_df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        self.model = self.build_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            output_shape=self.forecast_horizon
        )
        
        self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ],
            verbose=1
        )
        
        return self.evaluate(X_test, y_test)
    
    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)

        predictions_orig = self.scalers['units_sold'].inverse_transform(predictions)
        y_test_orig = self.scalers['units_sold'].inverse_transform(y_test)
        
        mape = np.mean(np.abs((y_test_orig - predictions_orig) / y_test_orig)) * 100
        rmse = np.sqrt(np.mean((y_test_orig - predictions_orig) ** 2))
        
        return {
            'MAPE': mape,
            'RMSE': rmse,
            'predictions': predictions_orig,
            'actual': y_test_orig
        }

    def forecast_future(self, current_data, periods=4):
        processed_data = self.prepare_data(current_data)
        X, _ = self.create_sequences(processed_data)
        
        predictions = self.model.predict(X[-1:])
        return self.scalers['units_sold'].inverse_transform(predictions)


forecaster = RetailDemandForecaster()
results = forecaster.fit('data.csv')
print(f"MAPE: {results['MAPE']:.2f}%")
print(f"RMSE: {results['RMSE']:.2f}")