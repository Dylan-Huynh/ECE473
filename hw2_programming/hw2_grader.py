#!/usr/bin/env python3

import graderUtil, collections, random

grader = graderUtil.Grader()
submission = grader.load('hw2_submission')


############################################################
# Problem 1a: Length of LCS with purely recursive algorithm

def testa():
    # Test around bases cases
    grader.requireIsEqual(1, submission.lcs_length("a", "ab", 1, 2))
    grader.requireIsEqual(2, submission.lcs_length("abc", "ac", 3, 2))
    grader.requireIsEqual(0, submission.lcs_length("", "ab", 0, 2))

grader.addBasicPart('1a-0-basic', testa, maxSeconds=1, description='simple test')

def testa(numChars, length):
    import random
    random.seed(42)
    # Generate a random string of the given length
    text1 = ''.join(chr(random.randint(ord('a'), ord('a') + numChars - 1)) for _ in range(length))
    text2 = ''.join(chr(random.randint(ord('a'), ord('a') + numChars - 1)) for _ in range(length))
    return submission.lcs_length(text1, text2, len(text1), len(text2))

grader.addHiddenPart('1a-0-hidden', lambda : testa(5, 5), maxPoints=1, maxSeconds=1, description='random trials')
grader.addHiddenPart('1a-1-hidden', lambda : testa(5, 10), maxPoints=1, maxSeconds=1, description='random trials (longer string)')
grader.addHiddenPart('1a-2-hidden', lambda : testa(10, 10), maxPoints=1, maxSeconds=1, description='random trials (more characters)')

############################################################
# Problem 1b: Length of LCS with top-down dynamic programming

def testb():
    # Test around bases cases
    grader.requireIsEqual(1, submission.lcs_length_dp("a", "ab", 1, 2))
    grader.requireIsEqual(0, submission.lcs_length_dp("", "ab", 0, 2))
    grader.requireIsEqual(3, submission.lcs_length_dp("daihhedrcs", "nbacghqtar", 10, 10))

grader.addBasicPart('1b-0-basic', testb, maxSeconds=1, description='simple test')

def testb(numChars, length):
    import random
    random.seed(42)
    # Generate a random string of the given length
    text1 = ''.join(chr(random.randint(ord('a'), ord('a') + numChars - 1)) for _ in range(length))
    text2 = ''.join(chr(random.randint(ord('a'), ord('a') + numChars - 1)) for _ in range(length))
    return submission.lcs_length_dp(text1, text2, len(text1), len(text2))

grader.addHiddenPart('1b-0-hidden', lambda : testb(10, 20), maxPoints=1, maxSeconds=1, description='random trials')
grader.addHiddenPart('1b-1-hidden', lambda : testb(20, 20), maxPoints=1, maxSeconds=1, description='random trials (longer string)')
grader.addHiddenPart('1b-2-hidden', lambda : testb(20, 30), maxPoints=1, maxSeconds=1, description='random trials (more characters)')

############################################################
# Problem 1c: Length of LCS with bottom-up dynamic programming
def testc():
    def get_result(s, t):
        table = [[-1] * (len(t) + 1) for _ in range(len(s) + 1)]
        submission.lcs_length_bu(s, t, table)
        return table[len(s)][len(t)]
    # Test around bases cases
    grader.requireIsEqual(1, get_result("a", "ab"))
    grader.requireIsEqual(0, get_result("", "ab"))
    grader.requireIsEqual(3, get_result("daihhedrcs", "nbacghqtar"))

grader.addBasicPart('1c-0-basic', testc, maxSeconds=1, description='simple test')

def testc(numChars, length):
    import random
    random.seed(42)
    def get_result(s, t):
        table = [[-1] * (len(t) + 1) for _ in range(len(s) + 1)]
        submission.lcs_length_bu(s, t, table)
        return table[len(s)][len(t)]
    # Generate a random string of the given length
    text1 = ''.join(chr(random.randint(ord('a'), ord('a') + numChars - 1)) for _ in range(length))
    text2 = ''.join(chr(random.randint(ord('a'), ord('a') + numChars - 1)) for _ in range(length))
    return get_result(text1, text2)

grader.addHiddenPart('1c-0-hidden', lambda : testc(10, 20), maxPoints=1, maxSeconds=1, description='random trials')
grader.addHiddenPart('1c-1-hidden', lambda : testc(20, 20), maxPoints=1, maxSeconds=1, description='random trials (longer string)')
grader.addHiddenPart('1c-2-hidden', lambda : testc(20, 30), maxPoints=1, maxSeconds=1, description='random trials (more characters)')

############################################################
# Problem 1d: Solution of LCS with top-down dynamic programming
def testd():
    # Test around bases cases
    grader.requireIsEqual("a", submission.lcs_solution_dp("a", "ab"))
    grader.requireIsEqual("", submission.lcs_solution_dp("", "ab"))
    grader.requireIsEqual("ahr", submission.lcs_solution_dp("daihhedrcs", "nbacghqtar"))

grader.addBasicPart('1d-0-basic', testd, maxSeconds=1, description='simple test')

def testd(numChars, length):
    import random
    random.seed(42)
    # Generate a random string of the given length
    text1 = ''.join(chr(random.randint(ord('a'), ord('a') + numChars - 1)) for _ in range(length))
    text2 = ''.join(chr(random.randint(ord('a'), ord('a') + numChars - 1)) for _ in range(length))
    return submission.lcs_solution_dp(text1, text2)

grader.addHiddenPart('1d-0-hidden', lambda : testd(10, 20), maxPoints=1, maxSeconds=1, description='random trials')
grader.addHiddenPart('1d-1-hidden', lambda : testd(20, 20), maxPoints=1, maxSeconds=1, description='random trials (longer string)')
grader.addHiddenPart('1d-2-hidden', lambda : testd(20, 30), maxPoints=1, maxSeconds=1, description='random trials (more characters)')


############################################################
# Problem 1e: Solution of LCS with bottom-up dynamic programming
def teste():
    # Test around bases cases
    grader.requireIsEqual("a", submission.lcs_solution_bu("a", "ab"))
    grader.requireIsEqual("", submission.lcs_solution_bu("", "ab"))
    grader.requireIsEqual("ahr", submission.lcs_solution_bu("daihhedrcs", "nbacghqtar"))

grader.addBasicPart('1e-0-basic', teste, maxSeconds=1, description='simple test')

def teste(numChars, length):
    import random
    random.seed(42)
    # Generate a random string of the given length
    text1 = ''.join(chr(random.randint(ord('a'), ord('a') + numChars - 1)) for _ in range(length))
    text2 = ''.join(chr(random.randint(ord('a'), ord('a') + numChars - 1)) for _ in range(length))
    return submission.lcs_solution_bu(text1, text2)

grader.addHiddenPart('1e-0-hidden', lambda : teste(10, 20), maxPoints=1, maxSeconds=1, description='random trials')
grader.addHiddenPart('1e-1-hidden', lambda : teste(20, 20), maxPoints=1, maxSeconds=1, description='random trials (longer string)')
grader.addHiddenPart('1e-2-hidden', lambda : teste(20, 30), maxPoints=1, maxSeconds=1, description='random trials (more characters)')

############################################################
# Problem 2: computeLongestPalindrome

def test2():
    # Test around bases cases
    grader.requireIsEqual(0, submission.computeLongestPalindromeLength(""))
    grader.requireIsEqual(1, submission.computeLongestPalindromeLength("a"))
    grader.requireIsEqual(2, submission.computeLongestPalindromeLength("aa"))
    grader.requireIsEqual(1, submission.computeLongestPalindromeLength("ab"))
    grader.requireIsEqual(3, submission.computeLongestPalindromeLength("animal"))
grader.addBasicPart('2-0-basic', test2, description='simple test')

def test2(numChars, length):
    import random
    random.seed(42)
    # Generate a random string of the given length
    text = ' '.join(chr(random.randint(ord('a'), ord('a') + numChars - 1)) for _ in range(length))
    ans2 = submission.computeLongestPalindromeLength(text)
grader.addHiddenPart('2-0-hidden', lambda : test2(2, 10), maxPoints=1, maxSeconds=1, description='random trials')
grader.addHiddenPart('2-1-hidden', lambda : test2(10, 10), maxPoints=1, maxSeconds=1, description='random trials (more characters)')
grader.addHiddenPart('2-2-hidden', lambda : test2(5, 20), maxPoints=1, maxSeconds=1, description='random trials')
grader.addHiddenPart('2-3-hidden', lambda : test2(5, 400), maxPoints=1, maxSeconds=1, description='random trials (longer)')

grader.grade()
