import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QTimer
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class LoginWidget(QWidget):
    def __init__(self, on_login_success):
        super().__init__() # Calls the QWidget constructor
        self.on_login_success, self.login_attempts = on_login_success, 0
        layout = QVBoxLayout(self)
        self.title_label = QLabel('CreditCredibility', alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)
        self.username_input = QLineEdit(placeholderText='Username') # Input field for username
        self.password_input = QLineEdit(placeholderText='Password', echoMode=QLineEdit.EchoMode.Password) # Input field for password
        self.recovery_input = QLineEdit(placeholderText='Recovery Key') # Input field for recovery key
        self.recovery_input.hide()
        self.message_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.login_button = QPushButton('Login', clicked=self.verify_login) # Button for logging in
        self.recovery_button = QPushButton('Enter', clicked=self.verify_recovery, objectName='recovery_button') # Button for entering the recovery key
        
        for w in [self.username_input, self.password_input, self.login_button,
                  self.message_label, self.recovery_input, self.recovery_button]:
            layout.addWidget(w)
            if isinstance(w, QPushButton): w.setFixedWidth(150)
        self.recovery_button.hide()

    def verify_login(self):
        with open(r"C:\Users\jmlka\Documents\compsci ia\login.txt") as file: # Open the login data file and read its contents
            login_data = file.read()
        if any(line.strip().split(":") == [self.username_input.text(), self.password_input.text()] # Check if entered username and password match any entry in the login data
               for line in login_data.splitlines()):
            self.message_label.setText('Identity Authenticated')
            QTimer.singleShot(1000, self.on_login_success)  # Delay before proceeding to success
        else:
            self.login_attempts += 1  # Increment login attempts on failure
            if self.login_attempts >= 3: # Prompt for recovery if too many failed attempts
                self.message_label.setText('Too many failed attempts. Use recovery key.')
                self.recovery_input.show()
                self.recovery_button.show()
                self.username_input.hide() # Hide login-related widgets
                self.password_input.hide()
                self.login_button.hide()
            else: # Show error message for incorrect credentials
                self.message_label.setText('Incorrect Username or Password')

    def verify_recovery(self):
        if self.recovery_input.text() == open(r"C:\Users\jmlka\Documents\compsci ia\recovery.txt").read().strip():
            self.login_attempts = 0
            self.message_label.setText('Account recovered. Please log in.')
            self.recovery_input.hide()
            self.recovery_button.hide()
            self.username_input.show() # Show the login-related widgets again
            self.password_input.show()
            self.login_button.show()
        else:
            self.message_label.setText('Incorrect recovery key')

class DataInputWidget(QWidget):
    def __init__(self, on_compute_success):
        super().__init__()
        self.on_compute_success = on_compute_success
        layout = QVBoxLayout(self)
        for text, func in [('Upload Customers CSV', lambda: self.handle_upload('customers')),
                           ('Upload New Customer CSV', lambda: self.handle_upload('new_customer')),
                           ('Compute Result', self.compute_result)]:
            layout.addWidget(QPushButton(text, clicked=func))

    def handle_upload(self, file_type):
        file_name, _ = QFileDialog.getOpenFileName(None, f"Upload {file_type.capitalize()} CSV", "", "CSV Files (*.csv)")
        if file_name:
            try:
                pd.read_csv(file_name).to_csv(f'{file_type}.csv', index=False)
                QMessageBox.information(self, "Success", f"{file_type.capitalize()} CSV uploaded successfully")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to upload {file_type.capitalize()} CSV: {str(e)}")

    def compute_result(self):
        if not all(os.path.exists(f) for f in ['customers.csv', 'new_customer.csv']):
            QMessageBox.warning(self, "Warning", "One or more files are missing")
            return
        
        old_customers = pd.read_csv('customers.csv')
        new_customer = pd.read_csv('new_customer.csv').iloc[0]
        
        categorical_features = ['EducationLevel', 'MaritalStatus', 'EmploymentStatus']
        numeric_features = ['Age', 'CreditScore', 'DebtInc', 'LoanSize']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='median'), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression())
        ])

        X = old_customers[categorical_features + numeric_features]
        y = old_customers['LoanRepaid']
        model.fit(X, y)

        new_X = pd.DataFrame([new_customer[categorical_features + numeric_features]])
        loan_repayment_prediction = "yes" if model.predict(new_X)[0] == 1 else "no"
        recommended_rate = round(model.predict_proba(new_X)[0][1] * 10, 2)
        
        mean_values = old_customers[numeric_features].mean()
        quantitative_diffs = {
            feature: {
                "new": (new_customer[feature] - mean_values[feature]) / mean_values[feature] * 100,
                "most_similar": None
            }
            for feature in numeric_features
        }

        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(old_customers[numeric_features])
        most_similar_index = knn.kneighbors(new_X[numeric_features])[1][0][0]
        most_similar_customer = old_customers.iloc[most_similar_index]

        for feature in numeric_features:
            quantitative_diffs[feature]["most_similar"] = (most_similar_customer[feature] - mean_values[feature]) / mean_values[feature] * 100
        
        self.storeCalculations = { # Stores calculations that will be later used to display graphs
            "most_similar_customer": most_similar_customer,
            "new_customer": new_customer,
            "old_customers": old_customers,
            "quantitative_diffs": quantitative_diffs,
            "numeric_features": numeric_features
        }
        self.on_compute_success(loan_repayment_prediction, recommended_rate, most_similar_customer, new_customer, old_customers, quantitative_diffs, self.storeCalculations)

class ResultWidget(QWidget):
    def __init__(self, loan_repayment_prediction, recommended_rate, most_similar_customer, 
                 on_display_graphs, new_customer, old_customers, quantitative_diffs, storeCalculations):
        super().__init__()
        layout = QVBoxLayout(self)
        self.title_label = QLabel('Statistics', alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)

        layout.addWidget(QLabel(f"Loan Repayment Prediction: {'Yes' if loan_repayment_prediction == 'yes' else 'No'}"))
        layout.addWidget(QLabel(f"Recommended Rate: {recommended_rate}%"))
        layout.addWidget(QLabel(f"Most Similar Customer: {most_similar_customer['Name']} (ID: {most_similar_customer['CustomerID']})"))

        categorical_features = ['EducationLevel', 'MaritalStatus', 'EmploymentStatus']
        comparison = "\n".join(
            [f"{feature}: \nNew={new_customer[feature]}\nMost Similar={most_similar_customer[feature]}\nOld Mode={old_customers[feature].mode().values[0]}\n"
             for feature in categorical_features])
        
        group_box = QGroupBox("Qualitative Comparison")
        group_layout = QVBoxLayout(group_box)
        comparison_label = QLabel(comparison)
        group_layout.addWidget(comparison_label)
        
        layout.addWidget(group_box)
        
        quantitative_comparison = "\n".join(
            [f"{feature}: New={quantitative_diffs[feature]['new']:.2f}%, Most Similar={quantitative_diffs[feature]['most_similar']:.2f}%"
             for feature in quantitative_diffs]
        )
        quantitative_group_box = QGroupBox("Quantitative Comparison")
        quantitative_group_layout = QVBoxLayout(quantitative_group_box)
        quantitative_comparison_label = QLabel(quantitative_comparison)
        quantitative_group_layout.addWidget(quantitative_comparison_label)
        
        layout.addWidget(quantitative_group_box)

        display_graphs_button = QPushButton('Display Graphs', clicked=on_display_graphs)
        layout.addWidget(display_graphs_button)

class CreditCredibilityApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('CreditCredibility')
        self.setGeometry(100, 100, 600, 400)
        self.main_layout = QVBoxLayout(self)
        self.login_widget = LoginWidget(on_login_success=self.show_data_input_page)
        self.main_layout.addWidget(self.login_widget)
        self.data_input_widget = None

    def show_data_input_page(self):
        self.login_widget.hide()
        if not self.data_input_widget:
            self.data_input_widget = DataInputWidget(on_compute_success=self.show_result_page)
            self.main_layout.addWidget(self.data_input_widget)
        self.data_input_widget.show()

    def show_result_page(self, loan_repayment_prediction, recommended_rate, most_similar_customer, 
    new_customer, old_customers, quantitative_diffs, storeCalculations):
        self.data_input_widget.hide()
        self.storeCalculations = storeCalculations
        result_widget = ResultWidget(
            loan_repayment_prediction, recommended_rate, most_similar_customer, self.show_graphs, 
            new_customer, old_customers, quantitative_diffs, self.storeCalculations
        )
        self.main_layout.addWidget(result_widget)

    def show_graphs(self):
        calc = self.storeCalculations
        
        numeric_features = calc['numeric_features']
        new_customer = calc['new_customer']
        old_customers = calc['old_customers']
        most_similar_customer = calc['most_similar_customer']
        
        fig, axes = plt.subplots(nrows=1, ncols=len(numeric_features), 
                                figsize=(len(numeric_features) * 5, 6))
        if len(numeric_features) == 1:
            axes = [axes]
        
        for i, feature in enumerate(numeric_features):
            ax = axes[i]
            new_value = new_customer[feature]
            old_mean = old_customers[feature].mean()
            similar_value = most_similar_customer[feature]
            diff_new = calc['quantitative_diffs'][feature]['new']
            diff_similar = calc['quantitative_diffs'][feature]['most_similar']
            
            ax.bar(['New'], [new_value], width=0.25, label='New Customer', color='b')
            ax.bar(['Old Mean'], [old_mean], width=0.25, label='Old Customers (Mean)', color='g')
            ax.bar(['Most Similar'], [similar_value], width=0.25, label='Most Similar Customer', color='r')
            
            ax.set_title(f'Comparison for {feature}')
            ax.set_ylabel('Value')
            ax.legend()

            ax.text(0, new_value, f'{diff_new:.2f}%', ha='center', va='bottom')
            ax.text(2, similar_value, f'{diff_similar:.2f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CreditCredibilityApp()
    ex.show()
    sys.exit(app.exec())

