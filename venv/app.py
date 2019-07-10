training_data = [
    ['Green', 3, 'Mango'],
    ['Yellow', 3, 'Mango'],
    ['Red', 3, 'Apple'],
    ['Green', 1, 'Grape'],
    ['Yellow', 3, 'Lemon']
]

# play?
'''training_data = [
    ["Sunny", "Hot", "High", False, "No"],
    ["Sunny", "Hot", "High", True, "No"],
    ["Overcast", "Hot", "High", False, "Yes"],
    ["Rainy", "Mild", "High", False, "Yes"],
    ["Rainy", "Cool", "Normal", False, "Yes"],
    ["Rainy", "Cool", "Normal", True, "No"],
    ["Overcast", "Cool", "Normal", True, "Yes"],
    ["Sunny", "Mild", "High", False, "No"],
    ["Sunny", "Cool", "Normal", False, "Yes"],
    ["Rainy", "Mild", "Normal", False, "Yes"],
    ["Sunny", "Mild", "Normal", True, "Yes"],
    ["Overcast", "Mild", "High", True, "Yes"],
    ["Overcast", "Hot", "Normal", False, "Yes"],
    ["Rainy", "Mild", "High", True, "No"],
]'''

#header = ['outlook', 'temperature', 'humidity', 'windy', 'label']
header = ['color', 'diameter', 'label']

def unique_vals(rows, col):
    return set([row[col] for row in rows])

def class_count(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value
    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value
    def __repr__(self):
        condition = '=='
        if is_numeric(self.value):
            condition = '>='
        return "Is %s %s %s?" % (header[self.column], condition, str(self.value))
def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def gini(rows):
    counts = class_count(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl]/float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

def info_gain(left, right, current_uncertainty):
    p = float(len(left))/(len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

def find_best_splits(rows):
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):
        values = set(row[col] for row in rows)
        for val in values:
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question

class Leaf:
    def __init__(self, rows):
        self.predictions = class_count(rows)

class Decision_Node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(rows):
    gain, question = find_best_splits(rows)
    if gain == 0:
        return Leaf(rows)
    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return Decision_Node(question, true_branch, false_branch)

def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return
    print(spacing + str(node.question))
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + " ")
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + " ")

def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl]/total*100)) + " %"
    return probs


my_tree = build_tree(training_data)
print_tree(my_tree)

testing_data =  [
        ['Green', 3, 'Mango'],
        ['Yellow', 4, 'Mango'],
        ['Red', 2, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Mango'],
    ]

'''testing_data =  [
        ["Sunny", "Cool", "High", True, "No"],
    ]'''

for row in testing_data:
    print("Actual: %s. vs Predicted: %s" % (row[-1], print_leaf(classify(row, my_tree))))