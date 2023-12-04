import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing")

class LogisticRegression:
    
    def __init__(self, k, n, method, alpha=0.001, max_iter=5000, use_ridge=False, ridge_lambda=0.1):
        self.k = k
        self.n = n
        self.alpha = alpha
        self.max_iter = max_iter
        self.method = method
        self.use_ridge = use_ridge
        self.ridge_lambda = ridge_lambda  # Ridge penalty parameter
        self.W = None
        self.losses = []

    def fit(self, X_train, y_train):
        self.W = np.random.rand(self.n, self.k)
        self.losses = []
                    
        if self.method == "batch":
            start_time = time.time()
            for i in range(self.max_iter):
                loss, grad = self.gradient(X_train, y_train)
                self.losses.append(loss)
                if self.use_ridge:
                    grad_penalty = self.ridge_lambda * self.W
                    grad += grad_penalty
                self.W = self.W - self.alpha * grad
                if i % 500 == 0:
                    print(f"Loss at iteration {i}: {loss}")
            print(f"Time taken: {time.time() - start_time} seconds")

        elif self.method == "minibatch":  # Add minibatch code here
            batch_size = 50  # Define batch size
            num_batches = len(X_train) // batch_size
            start_time = time.time()

            for i in range(self.max_iter):
                for batch_num in range(num_batches):
                    start = batch_num * batch_size
                    end = (batch_num + 1) * batch_size
                    batch_X = X_train[start:end]
                    batch_Y = y_train[start:end]

                    loss, grad = self.gradient(batch_X, batch_Y)
                    self.losses.append(loss)
                    if self.use_ridge:
                        grad_penalty = self.ridge_lambda * self.W
                        grad += grad_penalty
                    self.W = self.W - self.alpha * grad

                if i % 500 == 0:
                    print(f"Loss at iteration {i}: {loss}")

            print(f"Time taken: {time.time() - start_time} seconds")

        elif self.method == "sto":  # Add stochastic gradient descent (sto) code here
            start_time = time.time()
            list_of_used_ix = []

            for i in range(self.max_iter):
                idx = np.random.randint(X_train.shape[0])

                while idx in list_of_used_ix:
                    idx = np.random.randint(X_train.shape[0])

                X_train_sto = X_train[idx, :].reshape(1, -1)
                Y_train_sto = y_train[idx]

                loss, grad = self.gradient(X_train_sto, Y_train_sto)
                self.losses.append(loss)
                if self.use_ridge:
                    grad_penalty = self.ridge_lambda * self.W
                    grad += grad_penalty
                self.W = self.W - self.alpha * grad

                list_of_used_ix.append(idx)

                if len(list_of_used_ix) == X_train.shape[0]:
                    list_of_used_ix = []

                if i % 500 == 0:
                    print(f"Loss at iteration {i}: {loss}")

            print(f"Time taken: {time.time() - start_time} seconds")

        else:
            raise ValueError('Method must be one of the followings: "batch", "minibatch" or "sto".')

                    

    def accuracy(self, y_true, y_pred):
        correct_predictions = np.sum(y_true == y_pred)
        total_predictions = len(y_true)
        return correct_predictions / total_predictions

    def precision(self, y_true, y_pred, class_label):
        true_positive = np.sum((y_true == class_label) & (y_pred == class_label))
        false_positive = np.sum((y_true != class_label) & (y_pred == class_label))

        if true_positive + false_positive == 0:
            return 0  # To avoid division by zero

        return true_positive / (true_positive + false_positive)

    def recall(self, y_true, y_pred, class_label):
        true_positive = np.sum((y_true == class_label) & (y_pred == class_label))
        false_negative = np.sum((y_true == class_label) & (y_pred != class_label))

        if true_positive + false_negative == 0:
            return 0  # To avoid division by zero

        return true_positive / (true_positive + false_negative)

    def f1_score(self, y_true, y_pred, class_label):
        prec = self.precision(y_true, y_pred, class_label)
        rec = self.recall(y_true, y_pred, class_label)

        if prec + rec == 0:
            return 0  # To avoid division by zero

        return 2 * (prec * rec) / (prec + rec)

    def macro_precision(self, y_true, y_pred):
        unique_labels = np.unique(y_true)
        precision_sum = 0

        for class_label in unique_labels:
            precision_sum += self.precision(y_true, y_pred, class_label)

        return precision_sum / len(unique_labels)

    def macro_recall(self, y_true, y_pred):
        unique_labels = np.unique(y_true)
        recall_sum = 0

        for class_label in unique_labels:
            recall_sum += self.recall(y_true, y_pred, class_label)

        return recall_sum / len(unique_labels)

    def macro_f1_score(self, y_true, y_pred):
        unique_labels = np.unique(y_true)
        f1_score_sum = 0

        for class_label in unique_labels:
            f1_score_sum += self.f1_score(y_true, y_pred, class_label)

        return f1_score_sum / len(unique_labels)

    def weighted_precision(self, y_true, y_pred):
        unique_labels = np.unique(y_true)
        class_counts = [np.sum(y_true == label) for label in unique_labels]

        weighted_prec_sum = 0

        for label, class_count in zip(unique_labels, class_counts):
            weight = class_count / len(y_true)
            weighted_prec_sum += self.precision(y_true, y_pred, label) * weight

        return weighted_prec_sum

    def weighted_recall(self, y_true, y_pred):
        unique_labels = np.unique(y_true)
        class_counts = [np.sum(y_true == label) for label in unique_labels]

        weighted_rec_sum = 0

        for label, class_count in zip(unique_labels, class_counts):
            weight = class_count / len(y_true)
            weighted_rec_sum += self.recall(y_true, y_pred, label) * weight

        return weighted_rec_sum

    def weighted_f1_score(self, y_true, y_pred):
        unique_labels = np.unique(y_true)
        class_counts = [np.sum(y_true == label) for label in unique_labels]

        weighted_f1_sum = 0

        for label, class_count in zip(unique_labels, class_counts):
            weight = class_count / len(y_true)
            weighted_f1_sum += self.f1_score(y_true, y_pred, label) * weight

        return weighted_f1_sum

    def gradient(self, X, y):
        m = X.shape[0]
        h = self.h_theta(X, self.W)
        loss = -np.sum(y * np.log(h)) / m

        error = h - y
        grad_loss = X.T @ error

        if self.use_ridge:
            # Include Ridge penalty in the gradient
            grad_penalty = 2 * self.ridge_lambda * self.W
        else:
            grad_penalty = 0.0

        grad = grad_loss + grad_penalty
        return loss, grad

    def softmax(self, theta_t_x):
        return np.exp(theta_t_x) / np.sum(np.exp(theta_t_x), axis=1, keepdims=True)

    def softmax_grad(self, X, error):
        return X.T @ error

    def h_theta(self, X, W):
        return self.softmax(X @ W)

    def predict(self, X_test):
        return np.argmax(self.h_theta(X_test, self.W), axis=1)
    
    def _coef(self):
        return self.W[1:]

    def plot(self):
        plt.plot(np.arange(len(self.losses)), self.losses, label="Train Losses")
        plt.title("Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Losses")
        plt.legend()
        
    def plot_feature_importance(self, feature_names):
        feature_importance = np.abs(self._coef())
        sorted_indices = np.argsort(feature_importance)
        sorted_features = [feature_names[i] for i in sorted_indices]

        plt.figure(figsize=(10, 6))
        plt.barh(sorted_features, feature_importance[sorted_indices])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance')
        plt.show()
        
class RidgePenalty:
    def __init__(self, lambda_ridge):
        self.lambda_ridge = lambda_ridge

    def __call__(self, W):
        return self.lambda_ridge * np.sum(W**2)

    def derivative(self, W):
        return 2 * self.lambda_ridge * W

class RidgeLogisticRegression(LogisticRegression):
    def __init__(self, k, n, method, alpha=0.001, max_iter=5000, ridge_lambda=1.0):
        super().__init__(k, n, method, alpha, max_iter, use_ridge=True, ridge_lambda=ridge_lambda)
        self.penalty = RidgePenalty(ridge_lambda)

app = dash.Dash(__name__)

# Define styles for different elements
headline_style = {"fontSize": 24, "textAlign": "center", "color": "#FCF3CF"}
instruction_style = {"fontSize": 16, "textAlign": "center", "color": "#FEF9E7", "margin": "10px"}
input_style = {"width": "150px", "margin": "auto", "color": "#000000", "backgroundColor": "#FFFFFF"}
submit_button_style = {"textAlign": "center", "marginTop": "20px", "backgroundColor": "#F4D03F"}
car_price_style = {"fontSize": 20, "textAlign": "center", "marginTop": "20px", "color": "#FFFFFF"}

app.layout = html.Div(
    style={"backgroundColor": "#0B5345", "padding": "20px"},
    children=[
        html.H1("Car Price Prediction", style=headline_style),
        html.Div(
            "Fill in the values below to predict the range of the car price:",
            style=instruction_style,
        ),
        html.Div(
            [
                html.Label("Max power (bhp):", style={"color": "#FFFFFF"}),
                dcc.Input(id="max_power", type="number", style=input_style),
            ],
            style={"display": "block", "text-align": "center"},  # Center-align the label and input
        ),
        html.Div(
            [
                html.Label("Mileage (kmpl):", style={"color": "#FFFFFF"}),
                dcc.Input(id="mileage", type="number", style=input_style),
            ],
            style={"display": "block", "text-align": "center"},
        ),
        html.Div(
            [
                html.Label("Engine size (cc):", style={"color": "#FFFFFF"}),
                dcc.Input(id="engine", type="number", style=input_style),
            ],
            style={"display": "block", "text-align": "center"},
        ),
        html.Button("Submit", id="submit", style=submit_button_style),
        html.Div(id="car_price", style=car_price_style),

        
    ],
)

# Load the trained model
model = joblib.load('C:\\Users\\Panithi\\Desktop\\AIT\\DSAI\\ML\\Assignment 3\\app\\code\\Car-Price-3.model')

# Create a callback function to predict the car price and display it on the page
@app.callback(
    Output("car_price", "children"),
    [Input("submit", "n_clicks")],
    [State("max_power", "value"), State("mileage", "value"), State("engine", "value")],
)
def predict_car_price(n_clicks, max_power, mileage, engine_size):
    if n_clicks is None or n_clicks == 0:
        return ""

    input_data = [[max_power, mileage, engine_size]]
    input_data_scaled = model['scaler'].transform(input_data)
    intercept = np.ones((input_data_scaled.shape[0], 1))
    input_data_with_intercept = np.concatenate((intercept, input_data_scaled), axis=1)
    car_price = model['model'].predict(input_data_with_intercept)[0]

    sentences = [
        "For the group 0, the range of the predicted car price is between $0 to $2999.",
        "For the group 1, the range of the predicted car price is between $3000 to $200000.",
        "For the group 2, the range of the predicted car price is between $200001 to $400000.",
        "For the group 3, the predicted car price is higher than $400001."
    ]

    return html.Div([
        html.P(f"The range of the predicted car price is belong to the group {car_price}"),
        html.P("Here are some additional information:"),
        html.Ul(style={"list-style-type": "none"}, children=[html.Li(sentence) for sentence in sentences])
    ])

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
