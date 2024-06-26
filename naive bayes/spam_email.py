import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv('/home/phenx-07/Documents/Machine learning/naive bayes/spam.csv')
# print(df.head())
emails = [
    "Congratulations! You've won a free vacation to an exotic island! Click here to claim your prize now!",
    "Make money fast with our revolutionary new investment scheme! Guaranteed returns in just 7 days!",
    "Limited time offer: Get 50% off on all products! Don't miss out on this amazing deal!",
    "Hi John, I hope you're doing well. Attached is the report you requested. Let me know if you need any further assistance.",
    "Dear Customer, Thank you for your recent purchase. Your order has been successfully processed and will be shipped shortly."
]


desc = df.groupby('Category').describe()
print(desc)

df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0 )
print(df.head())

x_train,x_test,y_train,y_test = train_test_split(df['Message'],df['spam'] ,test_size=0.25)

v = CountVectorizer()
x_train_count  =v.fit_transform(x_train.values)
x_test_count = v.transform(x_test.values) 
print(x_train_count.toarray()[:3])

model = MultinomialNB()
model.fit(x_train_count,y_train)

emails_count = v.transform(emails)
accuracy = model.score(x_test_count, y_test)
print(accuracy)


predict = model.predict(emails_count)

print(predict)

