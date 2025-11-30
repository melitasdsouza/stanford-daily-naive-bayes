"""
classifier.py

Naive Bayes classifier for Stanford Daily headlines (News vs Opinion)

- Uses 40 News + 40 Opinion headlines for training
- Tests on 10 News + 10 Opinion held-out headlines
- Prints accuracy, confusion matrix, and distinctive word analysis
"""

import re
import math
from collections import Counter


# ---------------------------------------------------------------------
# 1. Raw data: 50 News headlines, 50 Opinion headlines
# ---------------------------------------------------------------------

news_headlines = [
    "AFDC and Lakeside to remain open during Thanksgiving break",
    "GSC votes to dissolve club sports funding umbrella",
    "Berkeley student founder shares company journey with Stanford entrepreneurs",
    "Faculty Senate debates student AI use",
    "Former government strategist discusses role of national policy in entrepreneurship",
    "Research over rivalry: Stanford and Berkeley collaborate to advance science and social impact",
    "UGS hears presentations on RA vacancies, Department of Public Safety initiatives",
    "Tree-ditions: Stanford rallies for 128th Big Game",
    "Former diplomat claims Obama, Trump failed on North Korea policy at CISAC event",
    "Andrew Luck pardons turkey in a new Big Game week tradition",
    "Police Blotter: Rape, offensive words, possession of controlled substances",
    "From Bearial to Regatta: LSJUMB prepares for Big Game",
    "Behind the scenes of Gaieties, the Stanford-Berkeley rivalry as a musical",
    "Stanford reacts to Nancy Pelosi's retirement from Congress",
    "StanfordNext project devises long-term campus growth initiative",
    "Class of 2029 enrollment data shows largest Asian, White percentage in three years",
    "Oski the Bear pronounced dead at Bearial",
    "Cross-campus couples turn rivalry to romance",
    "Food insecure students weather SNAP disruptions following government shutdown",
    "Dining hall dinner specials now served regularly",
    "Torrential rains batter Stanford on path through California",
    "GSC passes conditional collaboration with OCS, defers vote on club sports and natural gas",
    "Students find spaces to appreciate fall scenery across campus",
    "UGS affirms conditional collaboration with OCS, supports reinstatement of land acknowledgement",
    "Research Roundup: mini brains and mindful design",
    "Reconnect Stanford imagines a campus without social media",
    "Rice discusses restoring confidence in American democracy in Reimagining Democracy speaker series",
    "Levin and Rice discuss challenges for universities in era of misinformation",
    "Police Blotter: Identity theft, domestic violence and aggravated battery",
    "Stanford and City of Palo Alto hold Veterans Day celebration, recognize Vietnam War veteran",
    "Students to carry American flag from Death Valley to Mount Whitney, honoring veterans",
    "Global Health Opportunities Fair helps connect students to organizations in light of recent federal funding cuts",
    "RA vacancies leave understaffed residences confused and struggling",
    "Federal shutdown threatens students' Thanksgiving plans",
    "Congo Week x Stanford organizes for awareness and action",
    "Researchers discover new way to levitate cells",
    "GSE inaugurates first day of classes with dedication ceremony for new campus",
    "What daylight saving means for health productivity",
    "Stanford children's hospital to modernize after 25 million dollar donation",
    "Faculty Senate debates COLLEGE curriculum, votes to support post-bacc athlete admissions",
    # 10 News test headlines:
    "Chun Yang to remain in Old Union alongside future beer and wine service",
    "Students commemorate 1968 movement for ethnic studies programs with community art",
    "Stanford professor William Tarpeh named a MacArthur Fellow for wastewater treatment research",
    "GSC adopts bill to reinstate University land acknowledgements",
    "Faculty reflect on Dick Cheney's legacy",
    "UGS signs petition to reinstate University land acknowledgements",
    "Fifth annual Dine and Dialogue connects faculty and students across disciplines",
    "Santa Clara County sues Trump administration for restricting public service loan forgiveness",
    "Research Roundup: Accessing a healthier world through forests and fibers",
    "Women's athletic health conference presents breakthrough research for female athletes",
]

opinion_headlines = [
    "What Democrats can learn from Trump",
    "Faith and the ballot box",
    "The cathedral and the bazaar: Why Bay Area startup Big Game is its true genius",
    "Dear President Levin: Don't do a Berkeley",
    "Academic freedom on the line",
    "Why we condemn the politicized prosecution of the Stanford 11",
    "Policy may not be your major, but it governs your reality",
    "How Club Sports new funding policy hurts women",
    "Democracy Day needs a higher calling",
    "Beauty conformity at Stanford",
    "Inviting DeSantis to speak was easy. True free speech is hard.",
    "What 15 years of Daily opinion pieces reveal about diversity",
    "Stanford's place in a global tradition of student democratic participation",
    "Mamdani, Trump and Owning the Libs",
    "How Zohran Mamdani breaks the Model Minority myth",
    "What Sora 2 starring you means for Stanford",
    "Why Stanford was right to host Governor DeSantis",
    "How SLE humanizes Stanford's tech ethos",
    "No, Trump won't be America's Hitler. The reality is much more concerning",
    "The politics of mourning: To whom do we owe our remembrance?",
    "Charlie Kirk did not build genuine community. He tore it down.",
    "Why I am helping organize Faith and Freedom Night",
    "Stanford students would rather not think",
    "Why Stanford must support Prop 50",
    "Bigots like Ron DeSantis should not be platformed by Stanford",
    "No more softballs for strongmen",
    "Stanford's problem with blacking out",
    "We must refuse the Compact for Academic Excellence in Higher Education",
    "Avoiding the career funnel",
    "Talking through Zionism and anti-Zionism",
    "Affirm free speech, do not glorify terror",
    "What free expression discussion doesn't account for",
    "Consent is sexy",
    "Dear Class of 2000",
    "The crisis of free expression in the United States",
    "Now we'll see what President Levin is made of",
    "We are rising freshmen in a falling democracy",
    "A letter from the ASSU Executives",
    "Why study a foreign language?",
    "Tears, tumult and termination: How the Stanford hiring freeze affected someone who does not go here",
    # 10 Opinion test headlines:
    "Do we need an Opinions section?",
    "The time for collective action is now",
    "Better gun legislation starts with empathy",
    "On The Daily's lawsuit with FIRE",
    "What we're getting wrong about mental health and what neuroscience really tells us",
    "Vaping is only growing and our children will pay the price",
    "How we use LLMs matters",
    "We must build a united front to defend international students",
    "Stanford's Democratic duty to combat the MAGA rage",
    "Why America desperately needed Pope Leo XIV",
]


# ---------------------------------------------------------------------
# 2. Utilities
# ---------------------------------------------------------------------

def tokenize(text):
    """Lowercase and split headline into alphabetic tokens."""
    return re.findall(r"[a-z']+", text.lower())


def build_dataset():
    """Split into train/test."""
    train_news = news_headlines[:40]
    test_news = news_headlines[40:]

    train_op = opinion_headlines[:40]
    test_op = opinion_headlines[40:]

    train_data = [(tokenize(h), "News") for h in train_news] + \
                 [(tokenize(h), "Opinion") for h in train_op]

    test_data = [(tokenize(h), "News") for h in test_news] + \
                [(tokenize(h), "Opinion") for h in test_op]

    return train_data, test_data


# ---------------------------------------------------------------------
# 3. Naive Bayes Model
# ---------------------------------------------------------------------

class NaiveBayesHeadlineClassifier:
    def __init__(self):
        self.news_counts = Counter()
        self.op_counts = Counter()
        self.news_total = 0
        self.op_total = 0
        self.P_news = 0.5
        self.P_opinion = 0.5

    def train(self, data):
        n_news = n_op = 0
        for tokens, label in data:
            if label == "News":
                n_news += 1
                self.news_counts.update(tokens)
            else:
                n_op += 1
                self.op_counts.update(tokens)

        self.news_total = sum(self.news_counts.values())
        self.op_total = sum(self.op_counts.values())

        total = n_news + n_op
        self.P_news = n_news / total
        self.P_opinion = n_op / total

    def P_word_given_class(self, w, label):
        if label == "News":
            return self.news_counts[w] / self.news_total if self.news_counts[w] > 0 else 0.0
        else:
            return self.op_counts[w] / self.op_total if self.op_counts[w] > 0 else 0.0

    def classify(self, headline):
        tokens = tokenize(headline)
        logN = math.log(self.P_news)
        logO = math.log(self.P_opinion)

        for w in tokens:
            pN = self.P_word_given_class(w, "News")
            pO = self.P_word_given_class(w, "Opinion")
            if pN == 0 and pO == 0:
                continue
            logN += math.log(pN) if pN > 0 else -1e9
            logO += math.log(pO) if pO > 0 else -1e9

        return "News" if logN > logO else "Opinion"


# ---------------------------------------------------------------------
# 4. Evaluation helpers
# ---------------------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "of", "and", "to", "in", "on", "for", "with", "at",
    "by", "is", "are", "was", "were", "be", "been", "being", "this", "that",
    "from", "s", "t", "it", "as"
}

def confusion_matrix(preds, labels):
    tp_news = fn_news = tp_op = fn_op = 0
    for p, t in zip(preds, labels):
        if t == "News" and p == "News":
            tp_news += 1
        elif t == "News" and p == "Opinion":
            fn_news += 1
        elif t == "Opinion" and p == "Opinion":
            tp_op += 1
        elif t == "Opinion" and p == "News":
            fn_op += 1
    return [[tp_news, fn_news], [fn_op, tp_op]]


def top_distinct_words(main_counter, other_counter, k=10):
    result = []
    for w, c in main_counter.most_common():
        if w in STOPWORDS:
            continue
        if other_counter[w] > 0:
            continue
        result.append((w, c))
        if len(result) == k:
            break
    return result


# ---------------------------------------------------------------------
# 5. Main script
# ---------------------------------------------------------------------

def main():
    train_data, test_data = build_dataset()

    clf = NaiveBayesHeadlineClassifier()
    clf.train(train_data)

    # Predictions
    test_headlines = [" ".join(tokens) for tokens, _ in test_data]
    true_labels = [label for _, label in test_data]
    preds = [clf.classify(h) for h in test_headlines]

    # Accuracy
    correct = sum(p == t for p, t in zip(preds, true_labels))
    accuracy = correct / len(true_labels)
    print("\n=== RESULTS ===")
    print(f"Accuracy: {accuracy:.3f}")

    # Confusion matrix
    cm = confusion_matrix(preds, true_labels)
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print("[[News→News, News→Opinion],")
    print(" [Opinion→News, Opinion→Opinion]]")
    print(cm)

    # Distinctive words
    top_news = top_distinct_words(clf.news_counts, clf.op_counts)
    top_op = top_distinct_words(clf.op_counts, clf.news_counts)

    print("\nTop distinctive News words:")
    for w, c in top_news:
        print(f"  {w} ({c})")

    print("\nTop distinctive Opinion words:")
    for w, c in top_op:
        print(f"  {w} ({c})")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
