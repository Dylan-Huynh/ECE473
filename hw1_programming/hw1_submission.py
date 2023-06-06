import collections
import math
import re


## Note: these are for historical reasons called problem 3a-g but are assigned as problem 1
############################################################
# Problem 3a

def findAlphabeticallyLastWord(text):
    """
    Given a string |text|, return the word in |text| that comes last
    alphabetically (that is, the word that would appear last in a dictionary).
    A word is defined by a maximal sequence of characters without whitespaces.
    You might find max() and list comprehensions handy here.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    words = text.split()
    words.sort()
    return words[-1]
    # END_YOUR_CODE


############################################################
# Problem 3b

def euclideanDistance(loc1, loc2):
    """
    Return the Euclidean distance between two locations, where the locations
    are pairs of numbers (e.g., (3, 5)).
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)

    return math.sqrt((loc2[0] - loc1[0]) * (loc2[0] - loc1[0]) + (loc1[1] - loc2[1]) * (loc1[1] - loc2[1]))

    # return math.sqrt((loc1[0] - loc2[0]) ^ 2 + (loc1[1] - loc2[1]) ^ 2)
    # END_YOUR_CODE


############################################################
# Problem 3c

def mutateSentences(sentence):
    """
    Given a sentence (sequence of words), return a list of all "similar"
    sentences.
    We define a sentence to be similar to the original sentence if
      - it as the same number of words, and
      - each pair of adjacent words in the new sentence also occurs in the original sentence
        (the words within each pair should appear in the same order in the output sentence
         as they did in the original sentence.)
    Notes:
      - The order of the sentences you output doesn't matter.
      - You must not output duplicates.
      - Your generated sentence can use a word in the original sentence more than
        once.
    Example:
      - Input: 'the cat and the mouse'
      - Output: ['and the cat and the', 'the cat and the mouse', 'the cat and the cat', 'cat and the cat and']
                (reordered versions of this list are allowed)
    """
    # BEGIN_YOUR_CODE (our solution is 21 lines of code, but don't worry if you deviate from this)
    output = []
    diction = {}
    words = sentence.split()
    for i in range(len(words)):
        if i != 0:
            diction[words[i]] = words[i - 1]

    def route(dictionary, length, word, path):
        if path == length:
            similar = [word]
            return similar
        else:
            if word not in dictionary:
                return []
            else:
                similar = route(dictionary, length, dictionary[word], path + 1)
                if not similar:
                    return []
                similar.append(word)
                return similar

    for i in diction:
        sim = route(diction, len(words), i, 1)
        combination = ' '.join(sim)

        if sim and combination not in output:
            output.append(combination)
    return output
    # END_YOUR_CODE


############################################################
# Problem 3d

def sparseVectorDotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as collections.defaultdict(float), return
    their dot product.
    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    dot = 0
    for i in v1:
        if i in v2:
            dot += v1[i] * v2[i]

    return dot
    # END_YOUR_CODE


############################################################
# Problem 3e

def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)

    for i in v1:
        if i in v2:
            v1[i] += scale * v2[i]
    for i in v2:
        if i not in v1:
            v1[i] = scale * v2[i]

    return v1
    # END_YOUR_CODE


############################################################
# Problem 3f

def findSingletonWords(text):
    """
    Splits the string |text| by whitespace and returns the set of words that
    occur exactly once.
    You might find it useful to use collections.defaultdict(int).
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    splits = re.split('\s+', text)
    words = {}
    singleton = {}
    singleton = set()
    for i in splits:
        if i in words:
            words[i] += 1
        else:
            words[i] = 1
    for i in words:
        if words[i] == 1:
            singleton.add(i)

    return singleton

    # END_YOUR_CODE


############################################################
# Problem 3g

def computeLongestPalindromeLength(text):
    """A palindrome is a string that is equal to its reverse (e.g., 'ana').
    Compute the length of the longest palindrome that can be obtained by deleting
    letters from |text|.
    For example: the longest palindrome in 'animal' is 'ama'.
    Your algorithm should run in O(len(text)^2) time.
    You should first define a recurrence before you start coding.

    Hint: Let lpl(i,j) be the longest palindrome length of the substring text[i...j].

    Argue that lpl(i,j) = lpl(i+1, j-1) + 2 if text[i] == text[j], and
                          max{lpl(i+1, j), lpl(i, j-1)} otherwise
    (be careful with the boundary cases)


    FOR HOMEWORK 1, you may write a recursive version as described above.
    FOR HOMEWORK 2, you will have to write the non-recursive version described next, 
    which you can go ahead and do now if you understand how:
    
    Instead of writing a recursive function to find lpl (the most
    straightforward way of implementation has exponential running time
    due to repeated computation of the same subproblems), start by
    defining a 2-dimensional array which stores lpl(i,j), and fill it
    up in the increasing order of substring length.
    """
    # BEGIN_YOUR_CODE (our non-recursive solution is 13 lines of code, but don't worry if you deviate from this)
    def lpl(text, lb, ub):

        if (lb > ub):
            return 0
        if (lb == ub):
            return 1
        elif (text[lb] == text[ub] and lb + 1 == ub):
            return 2
        elif (text[lb] == text[ub]):
            return lpl(text, lb + 1, ub - 1) + 2

        return max(lpl(text, lb, ub - 1),
                   lpl(text, lb + 1, ub))

    print(text)
    longest = lpl(text, 0, len(text) - 1)
    print(longest)

    return longest
    # END_YOUR_CODE
