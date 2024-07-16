import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

nltk.download('vader_lexicon')
nltk.download('punkt')

# Conversation transcript with translations for key phrases
conversation = """
RA: नमस्ते श्री कुमार, मैं एक्स वाई जेड फाइनेंस से बोल रहा हूं। आपके लोन के बारे में बात करनी थी।
B: हां, बोलिए। क्या बात है?
RA: सर, आपका पिछले महीने का EMI अभी तक नहीं आया है। क्या कोई समस्या है?
(Sir, your EMI for last month has not been received yet. Is there any problem?)
B: हां, थोड़ी दिक्कत है। मेरी नौकरी चली गई है और मैं नया काम ढूंढ रहा हूं।
(Yes, there is a bit of a problem. I lost my job and am looking for new work.)
RA: ओह, यह तो बुरा हुआ। लेकिन सर, आपको समझना होगा कि लोन का भुगतान समय पर करना बहुत जरूरी है।
B: मैं समझता हूं, लेकिन अभी मेरे पास पैसे नहीं हैं। क्या कुछ समय मिल सकता है?
RA: हम समझते हैं आपकी स्थिति। क्या आप अगले हफ्ते तक कुछ भुगतान कर सकते हैं?
B: मैं कोशिश करूंगा, लेकिन पूरा EMI नहीं दे पाऊंगा। क्या आधा भुगतान चलेगा?
(I will try, but I won't be able to pay the full EMI. Will half the payment do?)
RA: ठीक है, आधा भुगतान अगले हफ्ते तक कर दीजिए। बाकी का क्या प्लान है आपका?
(Okay, make half the payment by next week. What is your plan for the rest?)
B: मुझे उम्मीद है कि अगले महीने तक मुझे नया काम मिल जाएगा। तब मैं बाकी बकाया चुका दूंगा।
(I hope to get a new job by next month. Then I will pay the remaining balance.)
RA: ठीक है। तो हम ऐसा करते हैं - आप अगले हफ्ते तक आधा EMI जमा कर दीजिए, और अगले महीने के 15 तारीख तक बाकी का भुगतान कर दीजिए। क्या यह आपको स्वीकार है?
(Okay. Let's do this - make half the EMI payment by next week and pay the remaining balance by the 15th of next month. Does this work for you?)
B: हां, यह ठीक रहेगा। मैं इस प्लान का पालन करने की पूरी कोशिश करूंगा।
(Yes, this will be fine. I will try my best to follow this plan.)
RA: बहुत अच्छा। मैं आपको एक SMS भेज रहा हूं जिसमें भुगतान की डिटेल्स होंगी। कृपया इसका पालन करें और समय पर भुगतान करें।
(Very good. I am sending you an SMS with the payment details. Please follow it and make the payment on time.)
B: ठीक है, धन्यवाद आपके समझने के लिए।
RA: आपका स्वागत है। अगर कोई और सवाल हो तो मुझे बताइएगा। अलविदा।
B: अलविदा।
"""


sia = SentimentIntensityAnalyzer()
sentences = sent_tokenize(conversation)

# each speaker
agent_sentiments = []
borrower_sentiments = []

for sentence in sentences:
    if sentence.startswith("RA:"):
        sentiment = sia.polarity_scores(sentence)
        agent_sentiments.append(sentiment)
    elif sentence.startswith("B:"):
        sentiment = sia.polarity_scores(sentence)
        borrower_sentiments.append(sentiment)

# average sentiment
def average_sentiment(sentiments):
    avg_sentiment = {
        'neg': 0,
        'neu': 0,
        'pos': 0,
        'compound': 0
    }
    for sentiment in sentiments:
        for key in avg_sentiment:
            avg_sentiment[key] += sentiment[key]
    for key in avg_sentiment:
        avg_sentiment[key] /= len(sentiments)
    return avg_sentiment

agent_avg_sentiment = average_sentiment(agent_sentiments)
borrower_avg_sentiment = average_sentiment(borrower_sentiments)

# Summary of conversation
summary = """
The recovery agent from XYZ Finance calls Mr. Kumar to discuss his overdue EMI payment.
Mr. Kumar explains he lost his job and is searching for new employment, which has led to his inability to make the payment.
The agent emphasizes the importance of timely payments but agrees to accept half of the EMI next week, with the remaining balance to be paid by the 15th of the next month.
Mr. Kumar agrees to this plan, and the agent promises to send him the payment details via SMS. Both parties end the call on a positive note.
"""

# Key Actions
actions = """
1. Mr. Kumar will make a half EMI payment by next week.
2. Mr. Kumar will pay the remaining balance by the 15th of the next month.
3. The recovery agent will send an SMS to Mr. Kumar with the payment details.
4. Mr. Kumar will follow the provided payment details and timelines.
"""

# Print results
print("Summary of Conversation:")
print(summary)

print("\nKey Actions or Next Steps Identified:")
print(actions)

print("\nSentiment Analysis of Recovery Agent:")
print(agent_avg_sentiment)

print("\nSentiment Analysis of Borrower (Mr. Kumar):")
print(borrower_avg_sentiment)
