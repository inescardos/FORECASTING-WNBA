
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.tree import plot_tree
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




def backwards_elimination(df, model, scaling=False):
    """Backwards elimination with SFS.
    Receives a dataframe and a model and returns the best features for the model.
    Also, it can scale the data if scaling is set to True.

    Args:
        df: the dataframe to be used
        model: the model to be used
        scaling (bool, optional): If the data needs scalling or not. Defaults to False.

    Returns:
        list: the list of the best features
    """
    scaler=MinMaxScaler()
    if(scaling):
        X = scaler.fit_transform(df.drop(columns=['playoff']))
        X = pd.DataFrame(X, columns=df.drop(columns=['playoff']).columns)
    else:
        X = df.drop(columns=['playoff'])
    Y = df['playoff']
    ffs = SFS(model, k_features=(3, 32), forward=False, n_jobs=-1, cv=0)
    ffs.fit(X, Y)
    features = list(ffs.k_feature_names_)
    teams_collumns_array=['tmID_ATL','tmID_CHA','tmID_CHI','tmID_CLE','tmID_CON','tmID_DET','tmID_HOU','tmID_IND','tmID_LAS','tmID_MIA','tmID_MIN','tmID_NYL','tmID_ORL','tmID_PHO','tmID_POR','tmID_SAC','tmID_SAS','tmID_SEA','tmID_UTA','tmID_WAS', 'tmID_TUL']

    #if it does not have the 'playoff', 'year', 'tmID', 'confID_EA', 'confID_WE' add it
    if 'playoff' not in features:
        features.append('playoff')
    if 'year' not in features:
        features.append('year')
    if 'confID_EA' not in features:
        features.append('confID_EA')
    if 'confID_WE' not in features:
        features.append('confID_WE')
    for team in teams_collumns_array:
        if team not in features and team in df.columns:
            features.append(team)
    return features


def select_k_best(df, model, scaling=False, years_back=3):
    """Selects the best k features for the model with mutual_info_classif.

    Args:
        df: the dataframe to be used
        model: the model to be used
        scaling (bool, optional): if the data needs scaling or not. Defaults to False.
        years_back (int, optional): number of years to train with. Defaults to 3.

    Returns:
        _type_: _description_
    """
    X=df.drop(['playoff'], axis=1)
    Y=df['playoff']
    hightest_accuracy_score = 0
    hightest_selected_column_names = []
    accuracy_scores = []
    teams_collumns_array=['tmID_ATL','tmID_CHA','tmID_CHI','tmID_CLE','tmID_CON','tmID_DET','tmID_HOU','tmID_IND','tmID_LAS','tmID_MIA','tmID_MIN','tmID_NYL','tmID_ORL','tmID_PHO','tmID_POR','tmID_SAC','tmID_SAS','tmID_SEA','tmID_UTA','tmID_WAS', 'tmID_TUL']
    for i in range(5, df.shape[1]):
        fs = SelectKBest(score_func=mutual_info_classif, k=i)
        # apply feature selection
        fs.fit_transform(X, Y)
        selected_feature_indices = fs.get_support()
        # Get the column names of the selected features
        selected_column_names = X.columns[selected_feature_indices]
        if 'playoff' not in selected_column_names:
            selected_column_names=selected_column_names.append(pd.Index(['playoff']))
        if 'year' not in selected_column_names:
            selected_column_names=selected_column_names.append(pd.Index(['year']))
        if 'confID_EA' not in selected_column_names:
            selected_column_names=selected_column_names.append(pd.Index(['confID_EA']))
        if 'confID_WE' not in selected_column_names:
            selected_column_names=selected_column_names.append(pd.Index(['confID_WE']))
        for team in teams_collumns_array:
            if team not in selected_column_names and team in df.columns:
                selected_column_names=selected_column_names.append(pd.Index([team]))
        accuracy_score, precision_score, recall_score, f1_score = train_evaluate_decision_tree_average(model, df[selected_column_names], scaling=scaling, years_back=years_back)
        accuracy_scores.append(accuracy_score)

        if(accuracy_score > hightest_accuracy_score):
            hightest_accuracy_score = accuracy_score
            hightest_selected_column_names = selected_column_names
    print("hightest_accuracy_score: ", hightest_accuracy_score)
        # Plotting the results
    plt.plot(range(5, df.shape[1]), accuracy_scores, marker='o')
    plt.title('Number of Features vs. Accuracy Score')
    plt.xlabel('Number of Features (K)')
    plt.ylabel('Accuracy Score')
    plt.show()
    return hightest_selected_column_names

def plot_metrics_over_time(years_tested, accuracy_scores, precision_scores, recall_scores, f1_scores, title="Normal Training"):
    """Plots the metrics over time of the accuracy, precision, recall and f1 scores.
    """
    # Create a graph to plot accuracy, precision, recall, and f1 over time
    plt.figure(figsize=(20, 5))
    plt.suptitle(title)

    # Plot accuracy
    plt.subplot(1, 4, 1)
    plt.plot(years_tested, accuracy_scores, marker='o')
    plt.title('Accuracy Over Time')
    plt.xlabel('Test Year')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Set the y-limits between 0.5 and 1

    # Plot precision
    plt.subplot(1, 4, 2)
    plt.plot(years_tested, precision_scores, marker='o', color='orange')
    plt.title('Precision Over Time')
    plt.xlabel('Test Year')
    plt.ylabel('Precision')
    plt.ylim(0, 1)  # Set the y-limits between 0.5 and 1

    # Plot recall
    plt.subplot(1, 4, 3)
    plt.plot(years_tested, recall_scores, marker='o', color='green')
    plt.title('Recall Over Time')
    plt.xlabel('Test Year')
    plt.ylabel('Recall')
    plt.ylim(0, 1)  # Set the y-limits between 0.5 and 1

    # Plot f1
    plt.subplot(1, 4, 4)
    plt.plot(years_tested, f1_scores, marker='o', color='red')
    plt.title('F1 Over Time')
    plt.xlabel('Test Year')
    plt.ylabel('F1')
    plt.ylim(0, 1)  # Set the y-limits between 0.5 and 1

    plt.tight_layout()
    plt.show()

    print(title)
    print("Accuracy: {:.2f}".format(
        sum(accuracy_scores) / len(accuracy_scores)), end=", ")
    print("Precision: {:.2f}".format(
        sum(precision_scores) / len(precision_scores)), end=", ")
    print("Recall: {:.2f}".format(
        sum(recall_scores) / len(recall_scores)), end=", ")
    print("F1: {:.2f}".format(sum(f1_scores) / len(f1_scores)))

    print("10 year")
    print("Accuracy:", accuracy_scores[-1], end=", ")
    print("Precision:", precision_scores[-1], end=", ")
    print("Recall:", recall_scores[-1], end=", ")
    print("F1:", f1_scores[-1])


def plot_metrics_over_time_test_train(years_tested, accuracy_scores, precision_scores, recall_scores, f1_scores, accuracy_train_scores, precision_train_scores, recall_train_scores, f1_train_scores, title="Normal Training"):
    """Plots the metrics over time of the accuracy, precision, recall and f1 scores of the test and train data.
    """
    # Create a graph to plot accuracy, precision, recall, and f1 over time
    # Define colors for the three training methods
    normal_color = 'blue'
    train_color = 'red'

    # Create separate plots for accuracy, precision, recall, and f1 over time
    plt.figure(figsize=(25, 5))
    plt.suptitle(title)

    # Create a small noise factor to offset the lines
    noise_factor = 0  # Adjust this value as needed

    # Plot accuracy for the three training methods with noise
    plt.subplot(1, 4, 1)
    plt.plot(years_tested, accuracy_scores + np.random.rand(len(years_tested))
             * noise_factor, marker='o', label='Test Results', color=normal_color)
    plt.plot(years_tested, accuracy_train_scores + np.random.rand(len(years_tested))
             * noise_factor, marker='o', label='Train results', color=train_color)
    plt.title('Accuracy Over Time')
    plt.xlabel('Test Year')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Set the y-limits between 0.5 and 1

    # Plot precision for the three training methods with noise
    plt.subplot(1, 4, 2)
    plt.plot(years_tested, precision_scores + np.random.rand(len(years_tested))
             * noise_factor, marker='o', label='Test Results', color=normal_color)
    plt.plot(years_tested, precision_train_scores + np.random.rand(len(years_tested))
             * noise_factor, marker='o', label='Train results', color=train_color)
    plt.title('Precision Over Time')
    plt.xlabel('Test Year')
    plt.ylabel('Precision')
    plt.ylim(0, 1)  # Set the y-limits between 0.5 and 1

    # Plot recall for the three training methods with noise
    plt.subplot(1, 4, 3)
    plt.plot(years_tested, recall_scores + np.random.rand(len(years_tested))
             * noise_factor, marker='o', label='Test Results', color=normal_color)
    plt.plot(years_tested, recall_train_scores + np.random.rand(len(years_tested))
             * noise_factor, marker='o', label='Train results', color=train_color)
    plt.title('Recall Over Time')
    plt.xlabel('Test Year')
    plt.ylabel('Recall')
    plt.ylim(0, 1)  # Set the y-limits between 0.5 and 1

    # Plot F1 for the three training methods with noise
    plt.subplot(1, 4, 4)
    plt.plot(years_tested, f1_scores + np.random.rand(len(years_tested))
             * noise_factor, marker='o', label='Test Results', color=normal_color)
    plt.plot(years_tested, f1_train_scores + np.random.rand(len(years_tested))
             * noise_factor, marker='o', label='Train results', color=train_color)
    plt.title('F1 Over Time')
    plt.xlabel('Test Year')
    plt.ylabel('F1')
    # Place the legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0, 1)  # Set the y-limits between 0.5 and 1

    plt.tight_layout()
    plt.show()

    print(title)
    print("Accuracy: {:.2f}".format(
        sum(accuracy_scores) / len(accuracy_scores)), end=", ")
    print("Precision: {:.2f}".format(
        sum(precision_scores) / len(precision_scores)), end=", ")
    print("Recall: {:.2f}".format(
        sum(recall_scores) / len(recall_scores)), end=", ")
    print("F1: {:.2f}".format(sum(f1_scores) / len(f1_scores)))

    print("10 year")
    print("Accuracy:", accuracy_scores[-1], end=", ")
    print("Precision:", precision_scores[-1], end=", ")
    print("Recall:", recall_scores[-1], end=", ")
    print("F1:", f1_scores[-1])


def plot_metrics_over_time_three(years_tested, accuracy_scores, precision_scores, recall_scores, f1_scores,
                                 bidirectional_accuracy_scores, bidirectional_precision_scores, bidirectional_recall_scores, bidirectional_f1_scores,
                                 kbest_accuracy_scores, kbest_precision_scores, kbest_recall_scores, kbest_f1_scores):
    """Plots the metrics over time of the accuracy, precision, recall and f1 scores of the three training methods.
    """
    # Define colors for the three training methods
    normal_color = 'blue'
    bidirectional_color = 'red'
    kbest_color = 'green'

    # Create separate plots for accuracy, precision, recall, and f1 over time
    plt.figure(figsize=(25, 5))
    plt.suptitle('Metrics Over Time', fontsize=16)

    # Create a small noise factor to offset the lines
    noise_factor = 0  # Adjust this value as needed

    # Plot accuracy for the three training methods with noise
    plt.subplot(1, 4, 1)
    plt.plot(years_tested, accuracy_scores + np.random.rand(len(years_tested))
             * noise_factor, marker='o', label='Normal Training', color=normal_color)
    plt.plot(years_tested, bidirectional_accuracy_scores + np.random.rand(len(years_tested))
             * noise_factor, marker='o', label='Directional Selection', color=bidirectional_color)
    plt.plot(years_tested, kbest_accuracy_scores + np.random.rand(len(years_tested))
             * noise_factor, marker='o', label='KBest Selection', color=kbest_color)
    plt.title('Accuracy Over Time')
    plt.xlabel('Test Year')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Set the y-limits between 0.5 and 1

    # Plot precision for the three training methods with noise
    plt.subplot(1, 4, 2)
    plt.plot(years_tested, precision_scores + np.random.rand(len(years_tested))
             * noise_factor, marker='o', label='Normal Training', color=normal_color)
    plt.plot(years_tested, bidirectional_precision_scores + np.random.rand(len(years_tested))
             * noise_factor, marker='o', label='Directional Selection', color=bidirectional_color)
    plt.plot(years_tested, kbest_precision_scores + np.random.rand(len(years_tested))
             * noise_factor, marker='o', label='KBest Selection', color=kbest_color)
    plt.title('Precision Over Time')
    plt.xlabel('Test Year')
    plt.ylabel('Precision')
    plt.ylim(0, 1)  # Set the y-limits between 0.5 and 1

    # Plot recall for the three training methods with noise
    plt.subplot(1, 4, 3)
    plt.plot(years_tested, recall_scores + np.random.rand(len(years_tested))
             * noise_factor, marker='o', label='Normal Training', color=normal_color)
    plt.plot(years_tested, bidirectional_recall_scores + np.random.rand(len(years_tested))
             * noise_factor, marker='o', label='Directional Selection', color=bidirectional_color)
    plt.plot(years_tested, kbest_recall_scores + np.random.rand(len(years_tested))
             * noise_factor, marker='o', label='KBest Selection', color=kbest_color)
    plt.title('Recall Over Time')
    plt.xlabel('Test Year')
    plt.ylabel('Recall')
    plt.ylim(0, 1)  # Set the y-limits between 0.5 and 1

    # Plot F1 for the three training methods with noise
    plt.subplot(1, 4, 4)
    plt.plot(years_tested, f1_scores + np.random.rand(len(years_tested)) *
             noise_factor, marker='o', label='Normal Training', color=normal_color)
    plt.plot(years_tested, bidirectional_f1_scores + np.random.rand(len(years_tested)) *
             noise_factor, marker='o', label='Directional Selection', color=bidirectional_color)
    plt.plot(years_tested, kbest_f1_scores + np.random.rand(len(years_tested))
             * noise_factor, marker='o', label='KBest Selection', color=kbest_color)
    plt.title('F1 Over Time')
    plt.xlabel('Test Year')
    plt.ylabel('F1')
    # Place the legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0, 1)  # Set the y-limits between 0.5 and 1

    plt.tight_layout()
    plt.show()

    print("Normal Training")
    print("Accuracy: {:.2f}".format(
        sum(accuracy_scores) / len(accuracy_scores)), end=", ")
    print("Precision: {:.2f}".format(
        sum(precision_scores) / len(precision_scores)), end=", ")
    print("Recall: {:.2f}".format(
        sum(recall_scores) / len(recall_scores)), end=", ")
    print("F1: {:.2f}".format(sum(f1_scores) / len(f1_scores)))

    print("10 year")
    print("Accuracy:", accuracy_scores[-1], end=", ")
    print("Precision:", precision_scores[-1], end=", ")
    print("Recall:", recall_scores[-1], end=", ")
    print("F1:", f1_scores[-1])

    print("Directional Selection")
    print("Accuracy: {:.2f}".format(sum(
        bidirectional_accuracy_scores) / len(bidirectional_accuracy_scores)), end=", ")
    print("Precision: {:.2f}".format(sum(
        bidirectional_precision_scores) / len(bidirectional_precision_scores)), end=", ")
    print("Recall: {:.2f}".format(
        sum(bidirectional_recall_scores) / len(bidirectional_recall_scores)), end=", ")
    print("F1: {:.2f}".format(
        sum(bidirectional_f1_scores) / len(bidirectional_f1_scores)))

    print("10 year")
    print("Accuracy:", bidirectional_accuracy_scores[-1], end=", ")
    print("Precision:", bidirectional_precision_scores[-1], end=", ")
    print("Recall:", bidirectional_recall_scores[-1], end=", ")
    print("F1:", bidirectional_f1_scores[-1])

    print("KBest Selection")
    print("Accuracy: {:.2f}".format(
        sum(kbest_accuracy_scores) / len(kbest_accuracy_scores)), end=", ")
    print("Precision: {:.2f}".format(
        sum(kbest_precision_scores) / len(kbest_precision_scores)), end=", ")
    print("Recall: {:.2f}".format(
        sum(kbest_recall_scores) / len(kbest_recall_scores)), end=", ")
    print("F1: {:.2f}".format(sum(kbest_f1_scores) / len(kbest_f1_scores)))

    print("10 year")
    print("Accuracy:", kbest_accuracy_scores[-1], end=", ")
    print("Precision:", kbest_precision_scores[-1], end=", ")
    print("Recall:", kbest_recall_scores[-1], end=", ")
    print("F1:", kbest_f1_scores[-1])


def split_and_train_conf_seperate(year, years_back, model, data, target_col="playoff", scaling=False, train_prob=False):
    """Splits the data into training and test sets and trains the model.

    Args:
        year: year to test on
        years_back: number of years to train with
        model: the model to be used
        data: the data to be used
        target_col: Defaults to "playoff".
        scaling (bool, optional): If the data needs scaling. Defaults to False.
        train_prob (bool, optional): If true, also calculate train probability. Defaults to False.

    Returns:
        y_test: the test target values
        y_pred: the predicted target values
        y_prob: the probability of the predicted target values
    """
    # Split the data into training and test sets
    train_data = data[data["year"] < year]
    train_data = train_data[train_data["year"] >= year - years_back]
    test_data = data[data["year"] == year]

    if (scaling):
        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data)
        train_data = pd.DataFrame(train_data, columns=data.columns)
        test_data = scaler.transform(test_data)
        test_data = pd.DataFrame(test_data, columns=data.columns)

    X_train = train_data.drop([target_col], axis=1)
    y_train = train_data[target_col]
    X_test = test_data.drop([target_col], axis=1)
    y_test = test_data[target_col]

    # PCA
    # pca = PCA(n_components=28, svd_solver='full')
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)

    # Create and train the decision tree model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    # y_pred = model.predict(X_test)
    # Make probability predictions on the test set
    # Sort the probabilities in reverse order and get the indices
    y_prob = model.predict_proba(X_test)

    sorted_indices = np.argsort(-y_prob[:, 1])
    # Set the top 4 predictions with target 1 and others with target 0
    y_pred = np.zeros_like(y_test)
    y_pred[sorted_indices[:4]] = 1

    if train_prob:
        train_data = train_data.copy()
        train_data.loc[:, 'y_prob'] = model.predict_proba(X_train)[:, 1]
        train_data['y_pred'] = 0
        years = train_data['year'].unique()
        for yr in years:
            yr_indices = train_data.index[train_data['year'] == yr].tolist()
            sorted_indices = np.argsort(
                -train_data['y_prob'].loc[train_data['year'] == yr].values).tolist()
            top_4_indices = [yr_indices[i] for i in sorted_indices[:4]]
            train_data.loc[top_4_indices, 'y_pred'] = 1
        return y_test, y_pred, y_prob[:, 1], y_train, train_data['y_pred'], train_data['y_prob']

    return y_test, y_pred, y_prob[:, 1]


def split_and_train_conferences_together(year, years_back, model, data, target_col="playoff", scaling=False, train_prob=False):
    """Splits the data into training and test sets and trains the model.

    Args:
        year: year to test on
        years_back: number of years to train with
        model: the model to be used
        data: the data to be used
        target_col: Defaults to "playoff".
        scaling (bool, optional): If the data needs scaling. Defaults to False.
        train_prob (bool, optional): If true, also calculate train probability. Defaults to False.

    Returns:
        y_test: the test target values
        y_pred: the predicted target values
        y_prob: the probability of the predicted target values
    """
    # Split the data into training and test sets
    train_data = data[data["year"] < year]
    train_data = train_data[train_data["year"] >= year - years_back]
    test_data = data[data["year"] == year]

    if (scaling):
        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data)
        train_data = pd.DataFrame(train_data, columns=data.columns)
        test_data = scaler.transform(test_data)
        test_data = pd.DataFrame(test_data, columns=data.columns)

    X_train = train_data.drop([target_col], axis=1)
    y_train = train_data[target_col]
    X_test = test_data.drop([target_col], axis=1)
    y_test = test_data[target_col]

    # PCA

    # pca = PCA(n_components=12, svd_solver='full')
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)

    # Create and train the decision tree model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    # y_pred = model.predict(X_test)
    # Make probability predictions on the test set and pick the top 4 from each conference
    y_prob = model.predict_proba(X_test)
    test_data = test_data.copy()
    test_data.loc[:, 'y_prob'] = y_prob[:, 1]
    test_data['y_pred'] = 0
    conferences = ['confID_EA', 'confID_WE']
    for conf_col in conferences:
        conf_indices = test_data.index[test_data[conf_col] == 1].tolist()
        sorted_indices = np.argsort(
            -test_data['y_prob'].loc[test_data[conf_col] == 1].values).tolist()
        top_4_indices = [conf_indices[i] for i in sorted_indices[:4]]
        test_data.loc[top_4_indices, 'y_pred'] = 1

    if train_prob:
        train_data = train_data.copy()
        train_data.loc[:, 'y_prob'] = model.predict_proba(X_train)[:, 1]
        train_data['y_pred'] = 0
        years = train_data['year'].unique()
        for yr in years:
            for conf_col in conferences:
                conf_indices = train_data.index[(train_data[conf_col] == 1) & (
                    train_data['year'] == yr)].tolist()
                sorted_indices = np.argsort(-train_data['y_prob'].loc[(
                    train_data[conf_col] == 1) & (train_data['year'] == yr)].values).tolist()
                top_4_indices = [conf_indices[i] for i in sorted_indices[:4]]
                train_data.loc[top_4_indices, 'y_pred'] = 1
        return y_test, test_data['y_pred'], y_prob[:, 1], y_train, train_data['y_pred'], train_data['y_prob']
    return y_test, test_data['y_pred'], y_prob[:, 1]


def split_data(data):
    """Splits the data into two dataframes, one for each conference."""
    data1 = data[data["confID_EA"] == 1]
    data2 = data[data["confID_WE"] == 1]
    return data1, data2


def train_evaluate_decision_tree_graph_conf_seperate(model, data, target_col="playoff", scaling=False, years_back=3, title="normal training"):
    """Trains and evaluates the model with the data for each year. Then plots the metrics over time.

    Args:
        model: the model to be used
        data: the data to be used
        target_col (str, optional): Defaults to "playoff".
        scaling (bool, optional): if the data needs scaling. Defaults to False.
        years_back (int, optional): years to train . Defaults to 3.
        title (str, optional): Defaults to "normal training".

    Returns:
        years_tested: the years tested
        accuracy_scores: the accuracy scores
        precision_scores: the precision scores
        recall_scores: the recall scores
        f1_scores: the f1 scores
    """
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    years_tested = []

    accuracy_scores_train = []
    precision_scores_train = []
    recall_scores_train = []
    f1_scores_train = []

    # Sort the data by the "year" column
    data = data.sort_values(by="year")
    data1, data2 = split_data(data)
    years = sorted(data["year"].unique())

    for year in years[2:]:

        y_test1, y_pred1, y_prob1, y_train_test1, y_train_pred1, y_train_prob1 = split_and_train(
            year, years_back, model, data1, target_col, scaling, train_prob=True)
        y_test2, y_pred2, y_prob2, y_train_test2, y_train_pred2, y_train_prob2 = split_and_train(
            year, years_back, model, data2, target_col, scaling, train_prob=True)

        # Merge of
        # Join y_test1 and y_test2
        y_test = np.concatenate([y_test1, y_test2])
        y_train_test = np.concatenate([y_train_test1, y_train_test2])
        # Join y_pred1 and y_pred2
        y_pred = np.concatenate([y_pred1, y_pred2])
        y_train_pred = np.concatenate([y_train_pred1, y_train_pred2])

        y_prob = np.concatenate([y_prob1, y_prob2])
        y_prob_train = np.concatenate([y_train_prob1, y_train_prob2])

        result_df = pd.DataFrame(
            {'y_test': y_train_test, 'y_pred': y_train_pred, 'y_prob': y_prob_train})

        # Calculate accuracy and precision
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        accuracy_train = accuracy_score(y_train_test, y_train_pred)
        precision_train = precision_score(y_train_test, y_train_pred)
        recall_train = recall_score(y_train_test, y_train_pred)
        f1_train = f1_score(y_train_test, y_train_pred)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        years_tested.append(year)

        accuracy_scores_train.append(accuracy_train)
        precision_scores_train.append(precision_train)
        recall_scores_train.append(recall_train)
        f1_scores_train.append(f1_train)


    plot_metrics_over_time_test_train(years_tested, accuracy_scores, precision_scores, recall_scores,
                                      f1_scores, accuracy_scores_train, precision_scores_train, recall_scores_train, f1_scores_train, title)
    return years_tested, accuracy_scores, precision_scores, recall_scores, f1_scores


def train_evaluate_decision_tree_graph(model, data, target_col="playoff", scaling=False, years_back=3, title="normal training", roc=False):
    """Trains and evaluates the model with the data for each year. Then plots the metrics over time.

    Args:
        model: the model to be used
        data: the data to be used
        target_col (str, optional): Defaults to "playoff".
        scaling (bool, optional): if the data needs scaling. Defaults to False.
        years_back (int, optional): years to train . Defaults to 3.
        title (str, optional): Defaults to "normal training".

    Returns:
        years_tested: the years tested
        accuracy_scores: the accuracy scores
        precision_scores: the precision scores
        recall_scores: the recall scores
        f1_scores: the f1 scores
    """
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    years_tested = []

    accuracy_scores_train = []
    precision_scores_train = []
    recall_scores_train = []
    f1_scores_train = []

    # Sort the data by the "year" column
    data = data.sort_values(by="year")

    years = sorted(data["year"].unique())

    for year in years[2:]:

        y_test, y_pred, y_prob, y_train_test, y_train_pred, y_train_prob = split_and_train_conferences_together(
            year, years_back, model, data, target_col, scaling, train_prob=True)

        # Calculate accuracy and precision
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        accuracy_train = accuracy_score(y_train_test, y_train_pred)
        precision_train = precision_score(y_train_test, y_train_pred)
        recall_train = recall_score(y_train_test, y_train_pred)
        f1_train = f1_score(y_train_test, y_train_pred)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        years_tested.append(year)

        accuracy_scores_train.append(accuracy_train)
        precision_scores_train.append(precision_train)
        recall_scores_train.append(recall_train)
        f1_scores_train.append(f1_train)

        if(year == 10 and roc):

            # Confusion matrix
            confusion_matrix_result = confusion_matrix(y_test, y_pred)
            confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix_result, display_labels=['0', '1']).plot()

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, label=f'Year {year} (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Receiver Operating Characteristic - Year {year}')
            plt.legend(loc="lower right")
            plt.show()

    plot_metrics_over_time_test_train(years_tested, accuracy_scores, precision_scores, recall_scores,
                                      f1_scores, accuracy_scores_train, precision_scores_train, recall_scores_train, f1_scores_train, title)
    return years_tested, accuracy_scores, precision_scores, recall_scores, f1_scores


def train_evaluate_decision_tree_average(model, data, target_col="playoff", scaling=False, title="normal training", years_back=3):
    """Trains and evaluates the model with the data for each year. Then plots the metrics over time.

    Args:
        model: the model to be used
        data: the data to be used
        target_col (str, optional): Defaults to "playoff".
        scaling (bool, optional): if the data needs scaling. Defaults to False.
        years_back (int, optional): years to train . Defaults to 3.
        title (str, optional): Defaults to "normal training".

    Returns:
        years_tested: the years tested
        accuracy_scores: the accuracy scores
        precision_scores: the precision scores
        recall_scores: the recall scores
        f1_scores: the f1 scores
    """
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    years_tested = []

    # Sort the data by the "year" column
    data = data.sort_values(by="year")
    data1, data2 = split_data(data)
    years = sorted(data["year"].unique())

    for year in years[2:]:

        y_test, y_pred, y_prob = split_and_train_conferences_together(
            year, years_back, model, data, target_col, scaling)

        # Calculate accuracy and precision
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        years_tested.append(year)

    return sum(accuracy_scores)/len(accuracy_scores), sum(precision_scores)/len(precision_scores), sum(recall_scores)/len(recall_scores), sum(f1_scores)/len(f1_scores)


def train_evaluate_decision_tree_years_back(model, data, target_col="playoff", scaling=False):
    """Trains and evaluates the model with the data for different years back. Then plots the metrics over time.

    Args:
        model (_type_): the model to be used
        data (_type_): the data to be used
        target_col (str, optional): _description_. Defaults to "playoff".
        scaling (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for year in range(1, 10):
        years_tested, accuracy, precision, recall, f1 = train_evaluate_decision_tree_graph(
            model, data, target_col, scaling, year, f"{year} years back")

        accuracy_scores.append(sum(accuracy)/len(accuracy))
        precision_scores.append(sum(precision)/len(precision))
        recall_scores.append(sum(recall)/len(recall))
        f1_scores.append(sum(f1)/len(f1))

    plot = plt.figure(figsize=(5, 5))
    plt.plot(range(1, 10), accuracy_scores, label="accuracy")
    plt.plot(range(1, 10), precision_scores, label="precision")
    plt.plot(range(1, 10), recall_scores, label="recall")
    plt.plot(range(1, 10), f1_scores, label="f1")
    plt.legend()
    plt.show()

    return years_tested, accuracy_scores, precision_scores, recall_scores, f1_scores
