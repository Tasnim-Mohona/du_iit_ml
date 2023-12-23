import streamlit as st
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader import load_dataset

# from ydata_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report

from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import Lasso, Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LassoCV
from sklearn.datasets import make_regression
from sklearn import linear_model
from sklearn.linear_model import RidgeCV

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold,StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn import datasets

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import plotly.express as px


st.set_page_config(layout="wide")


class DataPreprocessing:
    def __init__(self, df_or_path):
        if isinstance(df_or_path, pd.DataFrame):
            self.df = df_or_path
        elif isinstance(df_or_path, str):
            self.df = pd.read_csv(df_or_path, header=None)
        else:
            raise ValueError("Input must be a DataFrame or a file path.")

    def read_data(self):
        x = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]
        return x, y

    def split(self, x, y, split_size=80):
       
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=(100 - split_size), random_state=42, stratify=y, shuffle=True)
        # stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=(100 - split_size) / 100, random_state=42) 
        # train_index, test_index = next(stratified_splitter.split(x, y))
        # X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        # y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        return X_train, X_test, y_train, y_test

    def standardization(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled = pd.DataFrame(X_train_scaled)
        X_test_scaled = pd.DataFrame(X_test_scaled)

        st.write("X_train_scaled:")
        st.write(X_train_scaled)
        st.write("X_test_scaled:")
        st.write(X_test_scaled)

        return X_train_scaled, X_test_scaled

    @classmethod
    def discretize(cls, x, n_bins=5, strategy='uniform'):
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        discretized_data = discretizer.fit_transform(x)
        discretized_df = pd.DataFrame(discretized_data, columns=x.columns)
        return discretized_df

    def show_data(self, X_train, X_test, x, y):
        st.markdown('**1.2. Data splits**')
        st.write('Training set')
        st.info(X_train.shape)
        st.write('Test set')
        st.info(X_test.shape)

        st.markdown('**1.3. Features/Variable Details**:')
        st.write('X variable')
        st.info(list(x.columns))
        st.write('Y variable')
        st.info(y.name)

        
# Page 1: Data Upload and Preprocessing
def page_data_preprocessing():
    st.write("# A Comparative Analysis of Diverse Machine Learning Techniques in  Intrusion Detection Datasets")
    st.write("# Page 1: Data Upload and Preprocessing")

    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("## View of dataset")
        st.write(df.head())
        #DATA PREPROCESSING
        new_obj = DataPreprocessing(df)
        x, y = new_obj.read_data()
        X_train, X_test, y_train, y_test = new_obj.split(x, y)
        # new_obj.cross_validate()  # Add parentheses to invoke the method
        new_obj.show_data(X_train, X_test, x, y)  # Add parentheses to invoke the method
        X_train_scaled, X_test_scaled = new_obj.standardization(X_train, X_test)
        # Store data in session state
        st.session_state.data_processed = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
        }

    else:
        st.info('Awaiting for CSV file to be uploaded.')
        if st.button('Press to use Example Dataset'):
            # Using st.form to handle Enter key press
            with st.form(key='github_form'):
                # Text input for the GitHub link
                github_link = st.text_input("Enter GitHub link for the dataset:")
                submitted = st.form_submit_button('Load Dataset')

                if submitted and github_link:
                    # Load the dataset when the button is clicked with a valid link
                    df = load_dataset(github_link)
                    
                    st.markdown('The dataset from the provided GitHub link is used as the example.')
                    st.write(df.head(5))

                    # Rest of your code for data preprocessing
                    new_obj = DataPreprocessing(df)
                    x, y = new_obj.read_data()
                    X_train, X_test, y_train, y_test = new_obj.split(x, y)
                    new_obj.show_data(X_train, X_test, x, y)
                    X_train_scaled, X_test_scaled = new_obj.standardization(X_train, X_test)
                elif submitted and not github_link:
                    st.warning("Please enter a valid GitHub link.")



class Regularizer:
    def __init__(self, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled

    def lasso(self):
        lasso_cv = LassoCV(alphas=[0.1, 0.2, 0.3], cv=10, max_iter=11000)
        lasso_cv.fit(self.X_train, self.y_train)

        best_alpha = lasso_cv.alpha_
        st.write(f"Best alpha (Lasso): {best_alpha}")

        lasso_model = Lasso(alpha=best_alpha)
        lasso_model.fit(self.X_train_scaled, self.y_train)
        coefficients = lasso_model.coef_

        X_train_lasso = self.X_train_scaled * coefficients
        X_test_lasso = self.X_test_scaled * coefficients

        st.write("X_train_lasso:")
        st.write(X_train_lasso)
        st.write("X_test_lasso:")
        st.write(X_test_lasso)

        return X_train_lasso, X_test_lasso, coefficients

    def ridge(self):
        ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=10)
        ridge_cv.fit(self.X_train_scaled, self.y_train)

        ridge_best_alpha = ridge_cv.alpha_
        st.write(f"Best alpha (Ridge): {ridge_best_alpha}")

        ridge_model = Ridge(alpha=ridge_best_alpha)
        ridge_model.fit(self.X_train_scaled, self.y_train)
        coefficients = ridge_model.coef_

        X_train_ridge = self.X_train_scaled * coefficients
        X_test_ridge = self.X_test_scaled * coefficients

        st.write("X_train_ridge:")
        st.write(X_train_ridge)
        st.write("X_test_ridge:")
        st.write(X_test_ridge)

        return X_train_ridge, X_test_ridge

    def pls(self, n_components=7):
        pls_model = PLSRegression(n_components=n_components)
        pls_model.fit(self.X_train_scaled, self.y_train)

        X_train_pls = pls_model.transform(self.X_train_scaled)
        X_test_pls = pls_model.transform(self.X_test_scaled)

        imputer = SimpleImputer(strategy='mean')
        X_train_pls_imputed = imputer.fit_transform(X_train_pls)
        X_test_pls_imputed = imputer.transform(X_test_pls)

        train_feature_pls = pd.DataFrame(X_train_pls_imputed)
        test_feature_pls = pd.DataFrame(X_test_pls_imputed)

        st.write("train_feature_pls:")
        st.write(train_feature_pls)
        st.write("test_feature_pls:")
        st.write(test_feature_pls)
        return train_feature_pls, test_feature_pls

    @classmethod
    def discretize(cls, x, n_bins=5, strategy='uniform'):
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        discretized_data = discretizer.fit_transform(x)
        discretized_df = pd.DataFrame(discretized_data, columns=x.columns)
        return discretized_df


    def mutual_information(self, components):
        # Discretize data using the discretize method
        X_train_discretized = self.discretize(self.X_train_scaled)
        X_test_discretized = self.discretize(self.X_test_scaled)

        # Compute mutual information scores
        mi_scores = mutual_info_classif(X_train_discretized, self.y_train)

        # Convert the NumPy array to a Pandas Series
        mi_series = pd.Series(mi_scores)

        # Sort mutual information scores in descending order
        mi_sorted = mi_series.sort_values(ascending=False)

        # Select the top 'components' features
        selected_features = mi_sorted.head(components).index.tolist()
        

        # Get the selected features from the original X_train and X_test
        X_selected_train = self.X_train[selected_features].copy()
        X_selected_test = self.X_test[selected_features].copy()

        return X_selected_train, X_selected_test

class Classifier:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    def cross_validate_models(self, X, y, models, cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42)):
        scores_dict = {}
        for model_name, model in models:
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            scores_dict[model_name] = scores
        scores_df = pd.DataFrame(scores_dict)
        return scores_df
###NEW plot_roc_auc
   
    def plot_roc_auc(self, model, probabilities, title_suffix=""):
        roc_auc = roc_auc_score(self.y_test, probabilities)

        # ROC curve and AUC plot
        fpr, tpr, thresholds = roc_curve(self.y_test, probabilities)
        
        fig = px.line(x=fpr, y=tpr, labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                    title='Receiver Operating Characteristic (ROC) - {}'.format(title_suffix))

        # Add ROC curve
        fig.add_scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='navy', width=2, dash='dash'),
                        name='Random', showlegend=False)

        # Add AUC value as an annotation
        fig.add_annotation(x=0.5, y=0.5, text='AUC = {:.2f}'.format(roc_auc),
                        showarrow=False, font=dict(size=12, color='black'))

        # Show legend
        fig.update_layout(legend=dict(x=1, y=0, traceorder='normal', orientation='h'))

        # Display the plot using Streamlit
        st.plotly_chart(fig)
        st.title('Receiver Operating Characteristic (ROC) - {}'.format(title_suffix))

        return roc_auc

# ###RUNNING plot_roc_auc
#     def plot_roc_auc(self, model, probabilities, title_suffix=""):
#         roc_auc = roc_auc_score(self.y_test, probabilities)
#         fpr, tpr, thresholds = roc_curve(self.y_test, probabilities)

#         # Create a DataFrame for better display in Streamlit
#         roc_data = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr, 'Thresholds': thresholds})
#         st.title('Receiver Operating Characteristic (ROC) - {}'.format(title_suffix))
#         # Plot ROC curve using Plotly
#         fig = px.line(roc_data, x='False Positive Rate', y='True Positive Rate', labels={'False Positive Rate': 'False Positive Rate', 'True Positive Rate': 'True Positive Rate'},
#                     title='Receiver Operating Characteristic (ROC) - {}'.format(title_suffix))
#         # Add AUC information to the legend
#         auc_label = 'AUC = {:.2f}'.format(roc_auc)
#         fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
#         fig.add_annotation(x=0.5, y=0.5, text=auc_label, showarrow=False, font=dict(color='black'))
#         # Display the plot using Streamlit
        
#         st.plotly_chart(fig)
#         return roc_auc

    def knn(self):
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        knn_classifier.fit(self.X_train, self.y_train)
        predictions_knn = knn_classifier.predict(self.X_test)
        probabilities_knn = knn_classifier.predict_proba(self.X_test)[:, 1]

        precision = precision_score(self.y_test, predictions_knn)
        recall = recall_score(self.y_test, predictions_knn)
        accuracy = accuracy_score(self.y_test, predictions_knn)
        f1_measure = f1_score(self.y_test, predictions_knn)
        roc_auc = self.plot_roc_auc(knn_classifier, probabilities_knn, title_suffix="KNN Classification")
        result_dict = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'f-score': f1_measure, 'ROC-AUC': roc_auc}
        return result_dict

    def svm(self):
        svm_classifier = SVC(probability=True)
        svm_classifier.fit(self.X_train, self.y_train)
        predictions_svm = svm_classifier.predict(self.X_test)
        probabilities_svm = svm_classifier.predict_proba(self.X_test)[:, 1]

        precision = precision_score(self.y_test, predictions_svm)
        recall = recall_score(self.y_test, predictions_svm)
        accuracy = accuracy_score(self.y_test, predictions_svm)
        f1_measure = f1_score(self.y_test, predictions_svm)
        roc_auc = self.plot_roc_auc(svm_classifier, probabilities_svm, title_suffix="SVM Classification")
        result_dict = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'f-score': f1_measure, 'ROC-AUC': roc_auc}
        return result_dict

    def lda(self):
        lda_model = LinearDiscriminantAnalysis()
        lda_model.fit(self.X_train, self.y_train)
        predictions_lda = lda_model.predict(self.X_test)
        probabilities_lda = lda_model.predict_proba(self.X_test)[:, 1]

        precision = precision_score(self.y_test, predictions_lda)
        recall = recall_score(self.y_test, predictions_lda)
        accuracy = accuracy_score(self.y_test, predictions_lda)
        f1_measure = f1_score(self.y_test, predictions_lda)
        roc_auc = self.plot_roc_auc(lda_model, probabilities_lda, title_suffix="LDA Classification")

        result_dict = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'f-score': f1_measure, 'ROC-AUC': roc_auc}
        return result_dict

    def decision_tree(self):
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(self.X_train, self.y_train)
        predictions_decision_tree = decision_tree.predict(self.X_test)
        probabilities_decision_tree = decision_tree.predict_proba(self.X_test)[:, 1]

        precision = precision_score(self.y_test, predictions_decision_tree)
        recall = recall_score(self.y_test, predictions_decision_tree)
        accuracy = accuracy_score(self.y_test, predictions_decision_tree)
        f1_measure = f1_score(self.y_test, predictions_decision_tree)
        roc_auc = self.plot_roc_auc(decision_tree, probabilities_decision_tree, title_suffix="Decision Tree Classification")
        result_dict = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'f-score': f1_measure, 'ROC-AUC': roc_auc}
        return result_dict



    def plot_cross_val_boxplot(self, models, cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42)):
        cross_val_scores = self.cross_validate_models(self.X_train, self.y_train, models, cv=cv)

        # Transpose the labels
        labels = cross_val_scores.columns
        # Transpose the values
        values = [cross_val_scores[label].values for label in labels]

        # Flatten the list of arrays (values)
        flattened_values = np.concatenate(values)
        st.title('Cross-Validated Box Plot of Accuracy')
        # Create a box plot of the cross-validation scores using Plotly Express
        fig = px.box(x=np.repeat(labels, len(cross_val_scores)), y=flattened_values,
                    labels={'x': 'Model', 'y': 'Accuracy'}, title='Cross-Validation Box Plot')

        # Customize box plot appearance (change color to darkorange)
        for trace in fig.data:
            trace.marker.line.color = 'darkorange'
            trace.marker.line.width = 2

        # Add a horizontal line at y=0.5 for reference (you can customize this)
        fig.add_shape(type='line', x0=-1, x1=len(labels), y0=0.5, y1=0.5,
                    line=dict(color='navy', width=2, dash='dash'))

        # Display the plot using Streamlit
        st.plotly_chart(fig)
        

#####RUNNING BOX
    # def plot_cross_val_boxplot(self, models, cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42)):
    #     cross_val_scores = self.cross_validate_models(self.X_train, self.y_train, models, cv=cv)

    #     # Transpose the labels
    #     labels = cross_val_scores.columns
    #     # Transpose the values
    #     values = [cross_val_scores[label].values for label in labels]

    #     # Create a box plot of the cross-validation scores
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     ax.boxplot(values, labels=labels)
    #     ax.set_title('Cross-Validation Box Plot')
    #     ax.set_ylabel('Accuracy')

    #     # Display the plot using Streamlit
    #     st.pyplot(fig)
    #     st.title('Cross-Validation Box Plot')


    def plot_radar(self, results_df, title_suffix=""):
        categories = list(results_df.columns)
        models = results_df.index
        num_models = len(models)

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values = results_df.values.T

        values = np.concatenate((values, [values[:, 0]]), axis=1)
        angles += angles[:1]

        # Create a DataFrame for better display in Streamlit
        radar_data = pd.DataFrame(values, columns=categories, index=models)

        # Plot radar chart using Streamlit
        st.write("## Radar Plot - {}".format(title_suffix))
        st.line_chart(radar_data, use_container_width=True)
        st.plotly_chart(plt.figure(figsize=(8, 8)))  # Use plotly_chart to render the Matplotlib figure
        st.title('Radar Plot - {}'.format(title_suffix))
        st.legend(loc="upper right")



    def plot_scatter(self, results_dict, title_suffix=""):
        categories = list(results_dict.keys())
        values = list(results_dict.values())

        # Create a DataFrame for better display in Streamlit
        scatter_data = pd.DataFrame({'Metrics': categories, 'Values': values})

        # Scatter plot using Streamlit
        st.write("## Scatter Plot - {}".format(title_suffix))
        st.write(scatter_data.set_index('Metrics').plot(kind='scatter', x='Metrics', y='Values', color='blue', figsize=(8, 6)))
        st.xlabel('Metrics')
        st.ylabel('Values')



# Page 2: Run Regularizers
def page_run_regularizers():
    st.write("# Page 2: Run Regularizers")

    # Retrieve session state
    if 'data_processed' not in st.session_state:
        st.warning("Please upload and preprocess data on Page 1.")
        return

    # Accessing data from the session state
    X_train = st.session_state.data_processed['X_train']
    X_test = st.session_state.data_processed['X_test']
    y_train = st.session_state.data_processed['y_train']
    y_test = st.session_state.data_processed['y_test']
    X_train_scaled = st.session_state.data_processed['X_train_scaled']
    X_test_scaled = st.session_state.data_processed['X_test_scaled']

    # Regularizer option
    regularizer_option = st.selectbox('Select Regularizer', ['Select','Lasso', 'Ridge', 'PLS','Mutual Information'])

    if regularizer_option == 'Lasso':
        st.write("## Running Lasso Regularizer")
        lasso_model = Regularizer(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
        train_feature_lasso, test_feature_lasso, coefficients = lasso_model.lasso()

        # Total number of non-zero columns
        total_non_zero_train = (coefficients != 0).sum()

        st.write("\nTotal number of non-zero columns for training data:", total_non_zero_train)

        # Display Lasso features
        lasso_features = train_feature_lasso.iloc[:, coefficients != 0]
        st.write("Lasso Features:")
        st.write(lasso_features)

        st.markdown('**3.2. Regularization**')
        st.write('LASSO Regularized Dataset')
        st.info(train_feature_lasso.shape)
        st.write('Test set')
        st.info(test_feature_lasso.shape)

        st.markdown('**3.3. Lasso Features/Variable Details**:')
        st.write('Lasso Regularized Variables')
        st.info(list(lasso_features.columns))

        # Evaluate classifiers
        st.write("## Classifier Evaluation on Lasso Regularized Data")

        # Choose classifiers
        classifiers = st.multiselect('Select Classifiers to Evaluate', ['KNN', 'SVM', 'LDA', 'Decision Tree'])

        # Evaluate selected classifiers
        for classifier_name in classifiers:
            st.subheader(f"{classifier_name} Classification")
            classifier = Classifier(train_feature_lasso, test_feature_lasso, y_train, y_test)

            if classifier_name == 'KNN':
                result_dict = classifier.knn()
            elif classifier_name == 'SVM':
                result_dict = classifier.svm()
            elif classifier_name == 'LDA':
                result_dict = classifier.lda()
            elif classifier_name == 'Decision Tree':
                result_dict = classifier.decision_tree()
            st.title('Evaluation Metrics - {}'.format(classifier_name))
            st.write(result_dict)
            # JUPYTER X and y are your features and target variable, respectively
            clf = Classifier(train_feature_lasso,test_feature_lasso,y_train, y_test)

            models_to_cross_validate = [
                ('KNN',KNeighborsClassifier(n_neighbors=3)),
                ('SVM', SVC(probability=True)),
                ('LDA', LinearDiscriminantAnalysis()),
                ('Decision Tree', DecisionTreeClassifier())
            ]

            # Perform cross-validation and get scores
            cross_val_scores = clf.cross_validate_models(train_feature_lasso, y_train, models_to_cross_validate)

            # Plot cross-validation box plot
            clf.plot_cross_val_boxplot(models_to_cross_validate)
            # # Assuming you have instances of your Classifier class named knn, lda, decision_tree
            # svm = Classifier(train_feature_lasso,test_feature_lasso,y_train, y_test)
            # knn = Classifier(train_feature_lasso,test_feature_lasso,y_train, y_test)
            # lda = Classifier(train_feature_lasso,test_feature_lasso,y_train, y_test)
            # decision_tree = Classifier(train_feature_lasso,test_feature_lasso,y_train, y_test)
            # svm.svm()
            
            # # # Display Box Plot for Cross-Validation
            # # st.write(f"### {classifier_name} Cross-Validation Box Plot")
            # # models = [(classifier_name, getattr(classifier, f"{classifier_name.lower()}"))]
            # classifier.plot_cross_val_boxplot(models)


    elif regularizer_option == 'Ridge':
        st.write("## Running Ridge Regularizer")
        ridge_model = Regularizer(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
        train_feature_ridge, test_feature_ridge = ridge_model.ridge()

        # Perform further operations as needed...
        # Evaluate classifiers
        st.write("## Classifier Evaluation on Ridge Regularized Data")
        
        # Choose classifiers
        classifiers = st.multiselect('Select Classifiers to Evaluate', ['KNN', 'SVM', 'LDA', 'Decision Tree'])

        # Evaluate selected classifiers
        for classifier_name in classifiers:
            st.subheader(f"{classifier_name} Classification")
            classifier = Classifier(train_feature_ridge, test_feature_ridge, y_train, y_test)

            if classifier_name == 'KNN':
                result_dict = classifier.knn()
            elif classifier_name == 'SVM':
                result_dict = classifier.svm()
            elif classifier_name == 'LDA':
                result_dict = classifier.lda()
            elif classifier_name == 'Decision Tree':
                result_dict = classifier.decision_tree()
            st.title('Evaluation Metrics - {}'.format(classifier_name))
            st.write(result_dict)
            # Assuming you have instances of your Classifier class named knn, lda, decision_tree
            svm = Classifier(train_feature_ridge, test_feature_ridge,y_train, y_test)
            knn = Classifier(train_feature_ridge, test_feature_ridge,y_train, y_test)
            lda = Classifier(train_feature_ridge, test_feature_ridge,y_train, y_test)
            decision_tree = Classifier(train_feature_ridge, test_feature_ridge,y_train, y_test)

    elif regularizer_option == 'PLS':
        st.write("## Running PLS Regularizer")
        pls_model = Regularizer(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
        train_feature_pls, test_feature_pls = pls_model.pls()
        
        # Choose classifiers
        classifiers = st.multiselect('Select Classifiers to Evaluate', ['KNN', 'SVM', 'LDA', 'Decision Tree'])

        # Evaluate selected classifiers
        for classifier_name in classifiers:
            st.subheader(f"{classifier_name} Classification")
            classifier = Classifier(train_feature_pls, test_feature_pls, y_train, y_test)

            if classifier_name == 'KNN':
                result_dict = classifier.knn()
            elif classifier_name == 'SVM':
                result_dict = classifier.svm()
            elif classifier_name == 'LDA':
                result_dict = classifier.lda()
            elif classifier_name == 'Decision Tree':
                result_dict = classifier.decision_tree()
            st.title('Evaluation Metrics - {}'.format(classifier_name))
            st.write(result_dict)
            # Assuming you have instances of your Classifier class named knn, lda, decision_tree
            svm = Classifier(train_feature_pls, test_feature_pls,y_train, y_test)
            knn = Classifier(train_feature_pls, test_feature_pls,y_train, y_test)
            lda = Classifier(train_feature_pls, test_feature_pls,y_train, y_test)
            decision_tree = Classifier(train_feature_pls, test_feature_pls,y_train, y_test)
    

    elif regularizer_option == 'Mutual Information':
        st.write("## Running MI Regularizer")
        mi_model = Regularizer(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
        X_selected_train, X_selected_test = mi_model.mutual_information(components=6)
        


    # clf = Classifier(train_feature_pls, test_feature_pls, y_train, y_test)

    # models_to_cross_validate = [
    #     ('KNN', KNeighborsClassifier(n_neighbors=3)),
    #     ('SVM', SVC(probability=True)),
    #     ('LDA', LinearDiscriminantAnalysis()),
    #     ('Decision Tree', DecisionTreeClassifier())
    # ]

    # # Perform cross-validation and get scores
    # cross_val_scores = clf.cross_validate_models(train_feature_pls, y_train, models_to_cross_validate)

    # # Create a box plot of the cross-validation scores
    # st.pyplot(plt.boxplot(cross_val_scores.values, labels=cross_val_scores.keys()))
    # st.title('Cross-Validation Box Plot using PLS Regression')
    # st.ylabel('Accuracy')

    # # ROC-AUC plot for each model
    # for model_name, model in models_to_cross_validate:
    #     probabilities = model.fit(clf.X_train, clf.y_train).predict_proba(clf.X_test)[:, 1]
    #     st.pyplot(plt.figure(figsize=(8, 8)))
    #     clf.plot_roc_auc(model, probabilities, title_suffix=f'{model_name} - ROC-AUC')
    #     st.title(f'{model_name} - ROC-AUC')

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning App',
    layout='wide')

#---------------------------------#
# Model building
def build_model(df):
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Yratio

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100)
    
    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**1.3. Features/Variable Details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)

    rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
        random_state=parameter_random_state,
        max_features=parameter_max_features,
        criterion=parameter_criterion,
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf,
        bootstrap=parameter_bootstrap,
        oob_score=parameter_oob_score,
        n_jobs=parameter_n_jobs)
    rf.fit(X_train, Y_train)

    st.subheader('2. Model Performance')

    st.markdown('**2.1. Training set**')
    Y_pred_train = rf.predict(X_train)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_train, Y_pred_train) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_train, Y_pred_train) )

    st.markdown('**2.2. Test set**')
    Y_pred_test = rf.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_test, Y_pred_test) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_test, Y_pred_test) )

    st.subheader('3. Model Parameters')
    st.write(rf.get_params())




#---------------------------------#
st.write("""
# The Machine Learning App

In this implementation, the *RandomForestRegressor()* function is used in this app for build a regression model using the **Random Forest** algorithm.

Try adjusting the hyperparameters!

""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")


# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('2.1. Learning Parameters'):
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
    parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
    parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

with st.sidebar.subheader('2.2. General Parameters'):
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

with st.sidebar.header('3. Choose Regularizer'):
    regularizer_option = st.sidebar.selectbox('Select Regularizer', ['Select', 'Lasso', 'Ridge', 'Elastic Net', 'Least Squares', 'Mutual Information'])



# Page 2: Run Regularizers
def page_run_regularizers():
    st.write("# Page 2: Run Regularizers")

    # Retrieve session state
    if 'data_processed' not in st.session_state:
        st.warning("Please upload and preprocess data on Page 1.")
        return

    # Accessing data from the session state
    X_train = st.session_state.data_processed['X_train']
    X_test = st.session_state.data_processed['X_test']
    y_train = st.session_state.data_processed['y_train']
    y_test = st.session_state.data_processed['y_test']
    X_train_scaled = st.session_state.data_processed['X_train_scaled']
    X_test_scaled = st.session_state.data_processed['X_test_scaled']

# Page 3: Random Forest
# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. View of dataset**')
    st.write(df)
    build_model(df)

# else:
#     st.info('Awaiting for CSV file to be uploaded.')
#     if st.button('Press to use Example Dataset'):
#         # Boston housing dataset
#         boston = load_boston()
#         X = pd.DataFrame(boston.data, columns=boston.feature_names)
#         Y = pd.Series(boston.target, name='response')
#         df = pd.concat( [X,Y], axis=1 )

#         st.markdown('The Boston housing dataset is used as the example.')
#         st.write(df.head(5))

#         build_model(df)

# Main function
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Preprocessing", "Run Regularizers"])
    # Sidebar - Specify parameter settings

    if page == "Data Preprocessing":
        page_data_preprocessing()

    elif page == "Run Regularizers":
        page_run_regularizers()

if __name__ == "__main__":
    main()
