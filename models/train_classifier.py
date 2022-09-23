import sys
import re
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import nltk
import pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    # Load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(database_filepath, engine)
    # Split into X and y
    X = df['message']
    y = df[df.columns[4:]]
    category_names = list(y.columns)
    return X, y, category_names


def tokenize(text):
    # Remove urls - a check confirmed there are none in this dataset, but for
    # usability on future sets we add this code
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # Normalize text before tokenising
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # Tokenise
    words = word_tokenize(text)
    # Remove stop words 
    words = [w for w in words if w not in stopwords.words("english")]
    # Lemmatising
    words = [WordNetLemmatizer().lemmatize(w) for w in words]
    # Lemmatising verbs by specifiying the POS as verb
    words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]
    # Stemming
    words = [PorterStemmer().stem(w) for w in words]
    return words


def build_model():
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)), 
                     ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=0, n_estimators=200, max_depth=200)))])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    avg_precision = []
    avg_recall = []
    avg_fscore = []
    avg_support = []
    for num, column in enumerate(category_names):
        print(column)
        print(classification_report(Y_test[column], [ls[num] for ls in y_pred]))
        precision,recall,fscore,support=score(Y_test[column], [ls[num] for ls in y_pred], average='macro')
        avg_precision.append(precision)
        avg_recall.append(recall)
        avg_fscore.append(fscore)
        avg_support.append(support)
    print('Average of precisions is:', precision)
    print('Average of recalls is:', recall)
    print('Average of fscores is:', fscore)
    print('Average of supports is:', support)


def save_model(model, model_filepath):
    with open(model_filepath,"wb") as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()