using StringAnalysis
using DecisionTree
using SoleModels
using Random
using Statistics

# Set seed for reproducibility
Random.seed!(123)

# === 1. Dataset ===
# Training data
strs = [
    # Spam messages
    "buy cheap meds now",
    "limited time offer",
    "win money now!",
    "discount prices limited time",
    "viagra discount",
    "FREE GIFT waiting for you",
    "URGENT: Your account needs verification",
    "click here to claim prize",
    "act now and get 50% off",
    "congratulations you won 1000",
    "make money fast from home",
    "your loan has been approved",
    "special offer ends today",
    "increase your credit score instantly",
    "earn 5000 weekly working from home",
    "your payment has been declined",
    "hot singles in your area",
    "you have won the lottery",
    "investment opportunity high returns",
    "access to exclusive content click here",
    "lose weight fast guaranteed",
    "prescription drugs without prescription",
    "enlarge your manhood naturally",
    "we noticed suspicious activity on your account",
    "double your income in 30 days",
    "this is not a scam",
    "you've been selected for a survey",
    "get rich quick scheme",
    "bitcoin investment opportunity",
    "best rates on insurance",
    "consolidate your debt now",
    "free access to adult content",
    "your inheritance of 5,000,000",
    "unclaimed funds in your name",
    "easy cash loans no credit check",
    "meet local singles tonight",
    "your application has been approved",
    "lower your mortgage rates",
    "buy one get one free deal",
    "unlock your phone now",
    "amazing work opportunity",
    "make money online today",
    "get your degree online fast",
    "best prices on medications",
    "become a millionaire overnight",
    "get paid to take surveys",
    "refinance your home today",
    "important security alert",
    "your account will be suspended",
    "verify your identity now",
    "your payment was declined",
    "secret to making money online",
    "free trial membership",
    "claim your prize now",
    "risk-free investment opportunity",
    "remove your debt fast",
    "increase your website traffic",
    "luxury watches at low prices",
    "save big on your electric bill",
    "work from home business opportunity",

    # Ham messages
    "hey, are we still meeting?",
    "don't forget the meeting",
    "let's catch up soon",
    "please review the document",
    "can we talk later today",
    "when will you arrive?",
    "project meeting rescheduled",
    "dinner tonight at 7pm?",
    "thanks for your help yesterday",
    "can you send me the report?",
    "how was your weekend?",
    "let me know when you're free",
    "the presentation went well",
    "remind me about tomorrow's lunch",
    "do you need anything from the store?",
    "here are the files you requested",
    "see you in the morning",
    "let's discuss this in person",
    "what time is the conference call?",
    "happy birthday to you",
    "hope you're feeling better today",
    "the client was happy with our proposal",
    "just checking in on your progress",
    "did you receive my email from yesterday?",
    "please confirm your attendance",
    "looking forward to seeing you",
    "here's the information you asked for",
    "great job on the presentation",
    "can we reschedule our meeting?",
    "sorry I missed your call",
    "what did you think about the book?",
    "have a great weekend",
    "please forward this to the team",
    "I'll be running late today",
    "can you help me with this task?",
    "thoughts on the new proposal?",
    "I need your input on this project",
    "will you be home for dinner?",
    "remember to bring your laptop",
    "the deadline is approaching",
    "don't forget to sign the contract",
    "should we meet at the usual place?",
    "thanks for the update",
    "the boss wants to talk to you",
    "I've attached the document you needed",
    "please review before tomorrow",
    "are you available for a quick call?",
    "let's grab lunch next week",
    "how's the project coming along?",
    "I'll call you back in 5 minutes",
    "do you have time to talk today?",
    "please send me your thoughts",
    "I'll be out of office tomorrow",
    "remind me about that later",
    "did you get my message?",
    "what are your plans for the weekend?",
    "I'll need your help with this",
    "can you cover for me on Friday?",
    "I've updated the spreadsheet",
    "where should we meet?",
]

# Use string labels
labels = [
    # Spam labels (60 messages)
    "spam", "spam", "spam", "spam", "spam", "spam", "spam", "spam", "spam", "spam",
    "spam", "spam", "spam", "spam", "spam", "spam", "spam", "spam", "spam", "spam",
    "spam", "spam", "spam", "spam", "spam", "spam", "spam", "spam", "spam", "spam",
    "spam", "spam", "spam", "spam", "spam", "spam", "spam", "spam", "spam", "spam",
    "spam", "spam", "spam", "spam", "spam", "spam", "spam", "spam", "spam", "spam",
    "spam", "spam", "spam", "spam", "spam", "spam", "spam", "spam", "spam", "spam",

    # Ham labels (60 messages)
    "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam",
    "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam",
    "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam",
    "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam",
    "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam",
    "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam", "non spam"
]

# Convert to StringDocuments
texts = [StringDocument(f) for f in strs]

# === 2. Preprocessing: TF-IDF ===
train_corpus = Corpus(texts)
prepare!(train_corpus, strip_punctuation)  # preprocess training corpus
update_lexicon!(train_corpus)  # ensure lexicon is updated

# Save the lexicon from the corpus for later use with test documents
corpus_lexicon = lexicon(train_corpus)

# Create DTM and TF-IDF
dtm = DocumentTermMatrix(train_corpus)
tfidf_matrix = tf_idf(dtm)

# Extract feature vectors
features = [collect(values(tfidf_matrix[i])) for i in 1:length(texts)]
max_len = maximum(length.(features))
X = [vcat(f, zeros(max_len - length(f))) for f in features]
X_matrix = hcat(X...)'  # Each row is a document

# === 3. Train Random Forest ===
n_subfeatures = Int(round(sqrt(size(X_matrix, 2))))  # Square root of features is a good rule of thumb
n_trees = 10  # Increased number of trees for better performance
max_depth = 10  # Add a max depth to avoid overfitting

# Build the forest with improved parameters
model = build_forest(
    labels,  # Using string labels
    X_matrix,
    n_subfeatures,
    n_trees,
    0.7,  # proportion of samples to use for each tree
    max_depth
)

soleForest = solemodel(model)

# Print the forest structure
println("=== Random Forest Model ===")
println(soleForest)

# === 4. Predictions on test messages ===
test_messages = [
    # Likely spam
    "buy cheap medications online now",
    "congratulations! you've won 10,000",
    "limited time offer - 80% discount",
    "increase your credit score by 200 points",
    "unlock iphone without password",
    "work from home and earn 5000 weekly",
    "free gift card waiting for you",
    "your account has been compromised",
    "get rich quick with this method",
    "hot singles in your neighborhood",
    "make money while you sleep",
    "win a free vacation to hawaii",
    "urgent: your payment was declined",
    "lose 20 pounds in 2 weeks guaranteed",
    "best prices on luxury watches",

    # Likely ham
    "can we move the meeting to 3pm?",
    "please review the attached document",
    "are you available for lunch tomorrow?",
    "don't forget to bring your laptop",
    "I'll be out of office next week",
    "let me know what you think about the proposal",
    "thanks for your help with the project",
    "can you send me the latest version?",
    "the client meeting went well",
    "let's grab coffee sometime",
    "reminder about tomorrow's deadline",
    "would you like to join us for dinner?",
    "see you at the conference next month",
    "I've updated the spreadsheet with new data",
    "what are your thoughts on the feedback?"
]

# Convert test messages to StringDocuments
test_docs = [StringDocument(msg) for msg in test_messages]

# Create a test corpus
test_corpus = Corpus(test_docs)
prepare!(test_corpus, strip_punctuation)  # Apply same preprocessing

# Create Document-Term Matrix for test data using the vocabulary from training
test_dtm = DocumentTermMatrix(test_corpus)
test_tfidf = tf_idf(test_dtm)

# Extract feature vectors for test data
test_features = [collect(values(test_tfidf[i])) for i in 1:length(test_docs)]
test_max_len = size(X_matrix, 2)  # Use same feature dimensions as training data
test_X = [vcat(f, zeros(test_max_len - length(f))) for f in test_features]
test_X_matrix = hcat(test_X...)'  # Each row is a document

# Make predictions
predictions = apply_forest(model, test_X_matrix)

# Print predictions
println("\n=== Predictions ===")
for (i, (msg, pred)) in enumerate(zip(test_messages, predictions))
    # Print first 30 characters of message to make output more readable
    short_msg = length(msg) > 30 ? msg[1:30] * "..." : msg
    println("Message $(i): \"$(short_msg)\" => $(pred)")
end

# Calculate accuracy on the test set (assuming the first 15 are spam and last 15 are ham)
expected = vcat(fill("spam", 15), fill("non spam", 15))
correct = sum(predictions .== expected)
accuracy = correct / length(predictions)
println("\nAccuracy on test set: $(accuracy * 100)%")