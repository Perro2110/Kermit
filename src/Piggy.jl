# === 0. Setup ===
using StringAnalysis
using DecisionTree
using Random
using Statistics
using SparseArrays
using LinearAlgebra

println("Pacchetti caricati.")
Random.seed!(123)

# === 1. Dati ===
println("Definizione dati...")
strs = [
    # Spam (60)
    "buy cheap meds now", "limited time offer", "win money now!", "discount prices limited time", "viagra discount", "FREE GIFT waiting for you", "URGENT: Your account needs verification", "click here to claim prize", "act now and get 50% off", "congratulations you won 1000", "make money fast from home", "your loan has been approved", "special offer ends today", "increase your credit score instantly", "earn 5000 weekly working from home", "your payment has been declined", "hot singles in your area", "you have won the lottery", "investment opportunity high returns", "access to exclusive content click here", "lose weight fast guaranteed", "prescription drugs without prescription", "enlarge your manhood naturally", "we noticed suspicious activity on your account", "double your income in 30 days", "this is not a scam", "you've been selected for a survey", "get rich quick scheme", "bitcoin investment opportunity", "best rates on insurance", "consolidate your debt now", "free access to adult content", "your inheritance of 5,000,000", "unclaimed funds in your name", "easy cash loans no credit check", "meet local singles tonight", "your application has been approved", "lower your mortgage rates", "buy one get one free deal", "unlock your phone now", "amazing work opportunity", "make money online today", "get your degree online fast", "best prices on medications", "become a millionaire overnight", "get paid to take surveys", "refinance your home today", "important security alert", "your account will be suspended", "verify your identity now", "your payment was declined", "secret to making money online", "free trial membership", "claim your prize now", "risk-free investment opportunity", "remove your debt fast", "increase your website traffic", "luxury watches at low prices", "save big on your electric bill", "work from home business opportunity",
    # Ham (60)
    "hey, are we still meeting?", "don't forget the meeting", "let's catch up soon", "please review the document", "can we talk later today", "when will you arrive?", "project meeting rescheduled", "dinner tonight at 7pm?", "thanks for your help yesterday", "can you send me the report?", "how was your weekend?", "let me know when you're free", "the presentation went well", "remind me about tomorrow's lunch", "do you need anything from the store?", "here are the files you requested", "see you in the morning", "let's discuss this in person", "what time is the conference call?", "happy birthday to you", "hope you're feeling better today", "the client was happy with our proposal", "just checking in on your progress", "did you receive my email from yesterday?", "please confirm your attendance", "looking forward to seeing you", "here's the information you asked for", "great job on the presentation", "can we reschedule our meeting?", "sorry I missed your call", "what did you think about the book?", "have a great weekend", "please forward this to the team", "I'll be running late today", "can you help me with this task?", "thoughts on the new proposal?", "I need your input on this project", "will you be home for dinner?", "remember to bring your laptop", "the deadline is approaching", "don't forget to sign the contract", "should we meet at the usual place?", "thanks for the update", "the boss wants to talk to you", "I've attached the document you needed", "please review before tomorrow", "are you available for a quick call?", "let's grab lunch next week", "how's the project coming along?", "I'll call you back in 5 minutes", "do you have time to talk today?", "please send me your thoughts", "I'll be out of office tomorrow", "remind me about that later", "did you get my message?", "what are your plans for the weekend?", "I'll need your help with this", "can you cover for me on Friday?", "I've updated the spreadsheet", "where should we meet?",
]
labels = vcat(fill("spam", 60), fill("non spam", 60))
texts = StringDocument.(strs)
num_train_docs = length(texts)

# === 2. Training: Preprocessing & Feature Engineering ===
println("Preprocessing training...")
train_corpus = Corpus(texts)
prepare!(train_corpus, strip_punctuation)
update_lexicon!(train_corpus)
corpus_lexicon = lexicon(train_corpus)
num_terms = length(corpus_lexicon)
println("Lessico creato: $(num_terms) termini.")

dtm = DocumentTermMatrix(train_corpus)
println("Calcolo TF-IDF training...")
tfidf_result = tf_idf(dtm)

println("Conversione TF-IDF training a matrice densa e correzione orientamento...")
# Definisci le dimensioni attese nei due possibili orientamenti
expected_dims_doc_term = (num_train_docs, num_terms) # (docs x terms) - Target format
expected_dims_term_doc = (num_terms, num_train_docs) # (terms x docs) - Possible input format
actual_dims = size(tfidf_result)

if actual_dims == expected_dims_doc_term
    println("TF-IDF è già (docs x terms). Converto a denso...")
    global X_matrix = Matrix(tfidf_result)
elseif actual_dims == expected_dims_term_doc
    println("TF-IDF è (terms x docs). Converto a denso e traspongo...")
    # Prima converti a denso, poi usa ' per trasporre
    global X_matrix = Matrix(tfidf_result)'
else
    error("Dimensioni TF-IDF training ($(actual_dims)) non corrispondono né a $(expected_dims_doc_term) né a $(expected_dims_term_doc).")
end

# Verifica finale delle dimensioni di X_matrix
if size(X_matrix) != expected_dims_doc_term
     error("Errore! La matrice X_matrix finale ha dimensioni $(size(X_matrix)) invece di $(expected_dims_doc_term).")
end
println("Matrice features training (X_matrix) creata con dimensioni corrette: $(size(X_matrix))")

X_matrix[.!isfinite.(X_matrix)] .= 0.0

# === 3. Training Modello ===
println("Training Random Forest...")
n_trees = 3
max_depth = 10
model = build_forest(
    labels, X_matrix, Int(round(sqrt(num_terms))), n_trees, 0.7, max_depth
)
println("Modello addestrato.")

# === 4. Test: Preparazione & Feature Engineering ===
println("Preparazione dati test...")
test_messages = [
    # Spam (15)
    "buy cheap medications online now", "congratulations! you've won 10,000", "limited time offer - 80% discount", "increase your credit score by 200 points", "unlock iphone without password", "work from home and earn 5000 weekly", "free gift card waiting for you", "your account has been compromised", "get rich quick with this method", "hot singles in your neighborhood", "make money while you sleep", "win a free vacation to hawaii", "urgent: your payment was declined", "lose 20 pounds in 2 weeks guaranteed", "best prices on luxury watches",
    # Ham (15)
    "can we move the meeting to 3pm?", "please review the attached document", "are you available for lunch tomorrow?", "don't forget to bring your laptop", "I'll be out of office next week", "let me know what you think about the proposal", "thanks for your help with the project", "can you send me the latest version?", "the client meeting went well", "let's grab coffee sometime", "reminder about tomorrow's deadline", "would you like to join us for dinner?", "see you at the conference next month", "I've updated the spreadsheet with new data", "what are your thoughts on the feedback?"
]
expected_labels = vcat(fill("spam", 15), fill("non spam", 15))
num_test_docs = length(test_messages)

test_docs = StringDocument.(test_messages)
test_corpus = Corpus(test_docs)
prepare!(test_corpus, strip_punctuation)

test_dtm = DocumentTermMatrix(test_corpus, corpus_lexicon)
println("Calcolo TF-IDF test...")
test_tfidf_result = tf_idf(test_dtm)

println("Conversione TF-IDF test a matrice densa e correzione orientamento...")
# Definisci le dimensioni attese per il test set
expected_dims_doc_term_test = (num_test_docs, num_terms)
expected_dims_term_doc_test = (num_terms, num_test_docs)
actual_dims_test = size(test_tfidf_result)

if actual_dims_test == expected_dims_doc_term_test
    println("TF-IDF test è già (docs x terms). Converto a denso...")
    global test_X_matrix = Matrix(test_tfidf_result)
elseif actual_dims_test == expected_dims_term_doc_test
    println("TF-IDF test è (terms x docs). Converto a denso e traspongo...")
    global test_X_matrix = Matrix(test_tfidf_result)'
else
    error("Dimensioni TF-IDF test ($(actual_dims_test)) non corrispondono né a $(expected_dims_doc_term_test) né a $(expected_dims_term_doc_test).")
end

# Verifica finale delle dimensioni di test_X_matrix
if size(test_X_matrix) != expected_dims_doc_term_test
     error("Errore! La matrice test_X_matrix finale ha dimensioni $(size(test_X_matrix)) invece di $(expected_dims_doc_term_test).")
end
println("Matrice features test (test_X_matrix) creata con dimensioni corrette: $(size(test_X_matrix))")

test_X_matrix[.!isfinite.(test_X_matrix)] .= 0.0

# === 5. Predizione ===
println("Predizione sul test set...")
predictions = apply_forest(model, test_X_matrix)

# === 6. Valutazione Semplice ===
correct_predictions = sum(predictions .== expected_labels)
accuracy = correct_predictions / num_test_docs
println("\n--- Risultato ---")
println("Accuratezza sul test set: $(round(accuracy * 100, digits=2))% ($(correct_predictions)/$(num_test_docs))")
println("-----------------")

println("\nScript completato.")