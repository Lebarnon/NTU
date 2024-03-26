import unittest
import math
from questions import q1, q2, q3, q4, q5, q6, q7, q10
def hcfFunction(x, y):
    if x > y:
        smaller = y
    else:
        smaller = x
    for i in range(1, smaller+1):
        if((x % i == 0) and (y % i == 0)):
            hcf = i 
    return hcf

def encrypt(text,s):
    result = ""
    # transverse the plain text
    for i in range(len(text)):
        char = text[i]
        # Encrypt uppercase characters in plain text
        
        if (char.isupper()):
            result += chr((ord(char) + s-65) % 26 + 65)
        # Encrypt lowercase characters in plain text
        else:
            result += chr((ord(char) + s - 97) % 26 + 97)
    return result

class TestMethods(unittest.TestCase):

    def testQ1(self):
        self.assertEqual(q1(12, 8), 12%8)
        self.assertEqual(q1(15, 10), 15%10)
        self.assertEqual(q1(21, 14), 21%14)
        self.assertEqual(q1(0, 5), 0%5)
        self.assertEqual(q1(10, 1), 10%1)
        self.assertEqual(q1(24, 18), 24%18)
        self.assertEqual(q1(36, 27), 36%27)
        self.assertEqual(q1(48, 32), 48%32)
    
    def testQ2(self):
        self.assertEqual(q2(12, 8), hcfFunction(12, 8))
        self.assertEqual(q2(15, 10), hcfFunction(15, 10))
        self.assertEqual(q2(21, 14), hcfFunction(21, 14))
        self.assertEqual(q2(5, 1), hcfFunction(5, 1))
        self.assertEqual(q2(10, 1), hcfFunction(10, 1))
        self.assertEqual(q2(24, 18), hcfFunction(24, 18))
        self.assertEqual(q2(36, 27), hcfFunction(36, 27))
        self.assertEqual(q2(48, 32), hcfFunction(48, 32))
        self.assertEqual(q2(100, 75), hcfFunction(100, 75))
        self.assertEqual(q2(60, 45), hcfFunction(60, 45))
        self.assertEqual(q2(72, 54), hcfFunction(72, 54))
    
    def testQ3(self):
        self.assertEqual(q3(12, 8), hcfFunction(12, 8))
        self.assertEqual(q3(15, 10), hcfFunction(15, 10))
        self.assertEqual(q3(21, 14), hcfFunction(21, 14))
        self.assertEqual(q3(5, 1), hcfFunction(5, 1))
        self.assertEqual(q3(10, 1), hcfFunction(10, 1))
        self.assertEqual(q3(24, 18), hcfFunction(24, 18))
        self.assertEqual(q3(36, 27), hcfFunction(36, 27))
        self.assertEqual(q3(48, 32), hcfFunction(48, 32))
        self.assertEqual(q3(100, 75), hcfFunction(100, 75))
        self.assertEqual(q3(60, 45), hcfFunction(60, 45))
        self.assertEqual(q3(72, 54), hcfFunction(72, 54))

    def testQ4(self):
        self.assertEqual(q4(12, 8), hcfFunction(12, 8))
        self.assertEqual(q4(15, 10), hcfFunction(15, 10))
        self.assertEqual(q4(21, 14), hcfFunction(21, 14))
        self.assertEqual(q4(5, 1), hcfFunction(5, 1))
        self.assertEqual(q4(10, 1), hcfFunction(10, 1))
        self.assertEqual(q4(24, 18), hcfFunction(24, 18))
        self.assertEqual(q4(36, 27), hcfFunction(36, 27))
        self.assertEqual(q4(48, 32), hcfFunction(48, 32))
        self.assertEqual(q4(100, 75), hcfFunction(100, 75))
        self.assertEqual(q4(60, 45), hcfFunction(60, 45))
        self.assertEqual(q4(72, 54), hcfFunction(72, 54))

    def testQ5(self):
        self.assertEqual(q5(12, 8), math.lcm(12, 8))
        self.assertEqual(q5(15, 10), math.lcm(15, 10))
        self.assertEqual(q5(21, 14), math.lcm(21, 14))
        self.assertEqual(q5(5, 1), math.lcm(5, 1))
        self.assertEqual(q5(10, 1), math.lcm(10, 1))
        self.assertEqual(q5(24, 18), math.lcm(24, 18))
        self.assertEqual(q5(36, 27), math.lcm(36, 27))
        self.assertEqual(q5(48, 32), math.lcm(48, 32))
        self.assertEqual(q5(100, 75), math.lcm(100, 75))
        self.assertEqual(q5(60, 45), math.lcm(60, 45))
        self.assertEqual(q5(72, 54), math.lcm(72, 54))

    def testQ6(self):
        self.assertEqual(q6(12, 8, 3), math.lcm(12, 8, 3))
        self.assertEqual(q6(15, 10, 5), math.lcm(15, 10, 5))
        self.assertEqual(q6(21, 14, 7), math.lcm(21, 14, 7))
        self.assertEqual(q6(5, 1, 1), math.lcm(5, 1, 1))
        self.assertEqual(q6(10, 2, 1), math.lcm(10, 1, 2))
        self.assertEqual(q6(24, 18, 6), math.lcm(24, 18, 6))
        self.assertEqual(q6(36, 27, 9), math.lcm(36, 27, 9))
        self.assertEqual(q6(48, 32, 16), math.lcm(48, 32, 16))
        self.assertEqual(q6(100, 75, 25), math.lcm(100, 75, 25))
        self.assertEqual(q6(60, 45, 15), math.lcm(60, 45, 15))
        self.assertEqual(q6(72, 54, 7), math.lcm(72, 54, 7))
    
    def testQ7(self):
        self.assertEqual(q7("abc", 3), encrypt("abc", 3))
        self.assertEqual(q7("xyz", 3), encrypt("xyz", 3))
        self.assertEqual(q7("abc", 5), encrypt("abc", 5))
        self.assertEqual(q7("xyz", 5), encrypt("xyz", 5))
        self.assertEqual(q7("abc", 7), encrypt("abc", 7))
        self.assertEqual(q7("xyz", 7), encrypt("xyz", 7))
        self.assertEqual(q7("abc", 9), encrypt("abc", 9))
        self.assertEqual(q7("xyz", 9), encrypt("xyz", 9))
        self.assertEqual(q7("abc", 11), encrypt("abc", 11))
        self.assertEqual(q7("xyz", 11), encrypt("xyz", 11))
        self.assertEqual(q7("abc", 13), encrypt("abc", 13))
        self.assertEqual(q7("xyz", 13), encrypt("xyz", 13))
    def testQ10(self):
        self.assertEqual(q10(), True)

    



if __name__ == "__main__":
    unittest.main()
