import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=400, n_features=1, noise=20, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print(f"{'Degree':<10} {'Train MSE':<15} {'Test MSE':<15} {'R2':<10}")
print("-" * 55)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='gray', alpha=0.5, label='Test Data')

x_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
colors_list = ['blue', 'green', 'red', 'purple', 'orange']

for deg, color in zip(range(1, 6), colors_list):
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])

    pipe.fit(X_train, y_train)

    train_mse = mean_squared_error(y_train, pipe.predict(X_train))
    test_mse = mean_squared_error(y_test, pipe.predict(X_test))
    r2 = r2_score(y_test, pipe.predict(X_test))

    print(f"{deg:<10} {train_mse:<15.2f} {test_mse:<15.2f} {r2:<10.4f}")

    plt.plot(x_range, pipe.predict(x_range), color=color,
             label=f'Degree {deg}', linewidth=2)

plt.xlabel("Feature (Displacement proxy)")
plt.ylabel("MPG")
plt.title("Polynomial Regression: Degree Comparison")
plt.legend()
plt.tight_layout()
plt.show()
