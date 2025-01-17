from transformers import pipeline
import re

# Sample Conversation
conversation = """
Recovery Agent (RA): नमस्ते श्री कुमार, मैं एक्स वाई जेड फाइनेंस से बोल रहा हूं। आपके लोन के बारे में बात करनी थी।
Borrower (B): हां, बोलिए। क्या बात है?
RA: सर, आपका पिछले महीने का EMI अभी तक नहीं आया है। क्या कोई समस्या है?
B: हां, थोड़ी दिक्कत है। मेरी नौकरी चली गई है और मैं नया काम ढूंढ रहा हूं।
RA: ओह, यह तो बुरा हुआ। लेकिन सर, आपको समझना होगा कि लोन का भुगतान समय पर करना बहुत जरूरी है।
B: मैं समझता हूं, लेकिन अभी मेरे पास पैसे नहीं हैं। क्या कुछ समय मिल सकता है?
RA: हम समझते हैं आपकी स्थिति। क्या आप अगले हफ्ते तक कुछ भुगतान कर सकते हैं?
B: मैं कोशिश करूंगा, लेकिन पूरा EMI नहीं दे पाऊंगा। क्या आधा भुगतान चलेगा?
RA: ठीक है, आधा भुगतान अगले हफ्ते तक कर दीजिए। बाकी का क्या प्लान है आपका?
B: मुझे उम्मीद है कि अगले महीने तक मुझे नया काम मिल जाएगा। तब मैं बाकी बकाया चुका दूंगा।
RA: ठीक है। तो हम ऐसा करते हैं - आप अगले हफ्ते तक आधा EMI जमा कर दीजिए, और अगले महीने के 15 तारीख तक बाकी का भुगतान कर दीजिए। क्या यह आपको स्वीकार है?
B: हां, यह ठीक रहेगा। मैं इस प्लान का पालन करने की पूरी कोशिश करूंगा।
RA: बहुत अच्छा। मैं आपको एक SMS भेज रहा हूं जिसमें भुगतान की डिटेल्स होंगी। कृपया इसका पालन करें और समय पर भुगतान करें।
B: ठीक है, धन्यवाद आपके समझने के लिए।
RA: आपका स्वागत है। अगर कोई और सवाल हो तो मुझे बताइएगा। अलविदा।
B: अलविदा।
"""

# Preprocess conversation 
sentences = re.split(r'\n|(?<!\d)\.|\.(?!\d)', conversation)
sentences = [sent.strip() for sent in sentences if sent.strip()]

classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')


sentiment_analysis_results = [(sent, classifier(sent)) for sent in sentences]

# Display results
print("\nDetailed Sentiment Analysis:")
for sent, analysis in sentiment_analysis_results:
    sentiment = analysis[0]['label']
    print(f"- {sent.strip()} (Sentiment: {sentiment})")
