# Question 1
# In the version of the algorithm that uses subtraction, the calculation of the remainder (b = a mod b) is substituted with multiple subtractions.

def OriginalQ1(a, b):
    while a != b:
        if a < b:
            a = a - b
        else:
            b = b - a
    return b
    
def q1(a, b):
    while a >= b: # wrong
        if a > b: # wrong
            a = a - b
        else:
            a = b - a # wrong
    return a # wrong

# Question 2
# CF (highest common factor) or GCD (greatest common divisor) refer to the largest positive integer that divides both a, b without leaving a remainder. The Euclidean algorithm is based on the observation that if r is the remainder when a is divided by b, then the HCF of a and b is the same as the HCF of b and r.

def oringalQ2(a, b):
    while b != 0:
        c = a
        b = a % b
        a = b
    return a

def q2(a, b):
    while b != 0:
        c = b # wrong
        b = a % b
        a = c # wrong
    return a

# Question 3
# By repeatedly applying following identities, the Binary HCF algorithm simplifies the task of determining the HCF of two non-negative numbers a and b:
#     For instance, the HCF of 0 and a is simply a, since a is the greatest number that divides a.
#     Likewise, the HCF of b and 0 is b.
#     When both a and b are even, the HCF is twice the HCF of a/2 and b/2.
#     If b is even and a is odd, the HCF is the same as that of b/2 and a,
#     and if b is odd and a is even, the HCF is the same as that of b and a/2.
#     If both a and b are odd, the HCF equals the HCF of |b-a| and the smaller of b and a.

def originalQ3(a, b):
    if a == b:
        return a
    if a == 0:
        return b
    if b == 0:
        return a
    if (~b & 1) == 1: # check if even
        if (b & 1) == 1: # check if odd
            return originalQ3(a >> 1, b) # right shift a == divide a by 2
        else:
            return (originalQ3(a >> 1, b >> 1) << 1) # divide both a and b by 2 and multiply the result by 2
    if (~a & 1) == 1: # check if even
        return originalQ3(a, b >> 1) # divide b by 2
    if a < b: # check if a is smaller than b
        return originalQ3((a - b) >> 1, b) # divide the difference between a and b by 2
    return originalQ3((b - a) >> 1, a) # divide the difference between b and a by 2

def q3(a, b):
    if a == b:
        return a
    if a == 0:
        return b
    if b == 0:
        return a
    if (~b & 1) == 1: 
        if (a & 1) == 1: # wrong
            return q3(a, b >> 1) # wrong
        else:
            return (q3(a >> 1, b >> 1) << 1)
    if (~a & 1) == 1:
        return q3(a >> 1, b) # wrong
    if a > b: # wrong
        return q3((a - b) >> 1, b)
    return q3((b - a) >> 1, a)

# Question 4
# The brute force method for finding the HCF of two positive integers a, b involves selecting the smaller number between the two and checking every positive integer from 1 up to and including that number to see if it divides both a and b without leaving a remainder. The greatest value of i that satisfies this condition is the HCF of a and b.

def orginalQ4(a, b):
    hcf, i = 1, 1
    while i <= a and i <= b:
        if a % i == 0 or b % i == 0:
            hcf = i
        i = i + 1
    return i 

def q4(a, b):
    hcf, i = 1, 1
    while i <= a and i <= b:
        if a % i == 0 and b % i == 0: # wrong
            hcf = i
        i = i + 1
    return hcf # wrong

# Question 5
# The LCM (Least Common Multiple) of two numbers a, b refers to the minimum number that is evenly divisible by both of the given numbers. This implementation of finding the LCM of a, b using a conditional for loop is to first determine which number is greater, and then set it as the starting point for the loop. From there, it can iterate through the multiples of this number and check if each multiple is also divisible by the other number. Once it identifies a multiple that satisfies both conditions, that number is the LCM of a, b.

def originalQ5(a, b):
    x = max(a, b)
    y = min(a, b)
    for i in range(y, a * b + 1, x):
        if i % x == 0:
            return i

def q5(a, b):
    x = max(a, b)
    y = min(a, b)
    for i in range(x, a * b + 1, x): # wrong
        if i % y == 0: # wrong
            return i

# Question 6
# To find the LCM of three numbers a, b, c using HCF are shown below. It first finds the HCF of any two of the given numbers. Then divide each of the given numbers by the HCF and multiply the HCF by all of the obtained quotients and any remaining numbers that have not been factored out. The resulting product will give the LCM of all the given numbers.
        
def originalQ6(a, b, c):
    def hcf(a, b):
        while b:
            c = b
            b = a % b
            a = c
        return a
    def lcm2(a, b):
        c = b // hcf(a, b)
        return c * b
    def lcm3(a, b, c):
        x = a * hcf(b, c)
        y = hcf(a, lcm2(b, c))
        return x // y   
    return lcm3(a,b,c)

def q6(a, b, c):
    def hcf(a, b):
        while b:
            c = b
            b = a % b
            a = c
        return a
    def lcm2(a, b):
        c = b // hcf(a, b)
        return c * a # wrong
    def lcm3(a, b, c):
        x = a * lcm2(b, c) # wrong
        y = hcf(a, lcm2(b, c))
        return x // y 
    return lcm3(a,b,c)


# Question 7
# The Caesar Cipher is a straightforward encryption technique used to shift the letters in a piece of text by a certain number of positions in the alphabet. While implementing this cipher, it's crucial to ensure that the encryption and decryption correctly handle both uppercase and lowercase letters and wrap around the alphabet. However, mistakes in the implementation can lead to incorrect encryption or decryption outcomes.
def originalQ7(text, shift):
    encrypted_text = ""
    for char in text:
        if char.isupper():
            shifted_char = chr((ord(char) + shift - 65) % 26 + 65)
            encrypted_text += shifted_char
        else:
            encrypted_text += char
    return encrypted_text

def q7(text, shift):
    encrypted_text = ""
    for char in text:
        if char.isupper():
            shifted_char = chr((ord(char) + shift - 65) % 26 + 65)
            encrypted_text += shifted_char
        else:
            encrypted_text += chr((ord(char) + shift - 97) % 26 + 97) # wrong
    return encrypted_text

# Question 8 
# The Diffie-Hellman key exchange algorithm allows two parties to securely share a secret key over an unsecured communication channel. This shared key can then be used for secure communication between the two.

def originalQ8(p, g, private_a, private_b):
    public_a = g ** private_a % p
    public_b = g ** private_b % p
    shared_secret_by_alice = public_a ** private_b % p
    shared_secret_by_bob = public_b ** private_a % p
    return shared_secret_by_alice #Returning Alice's shared secret

def q8(p, g, private_a, private_b):
    public_a = g ** private_a % p
    public_b = g ** private_b % p
    shared_secret_by_alice = public_b ** private_a % p # wrong
    shared_secret_by_bob = public_a ** private_b % p # wrong
    return shared_secret_by_alice #Returning Alice's shared secret


# Question 9
# RSA key generation is the first step in setting up RSA encryption. It involves selecting two large prime numbers, calculating their product to get the modulus n, and finding two key exponents, e and d, that satisfy certain mathematical properties.

import random
import sympy
def originalQ9():
    p = 2
    q = 2
    n = p * q
    phi = (p-1) * (q-1)
    e = e = random.choice([i for i in range(2, phi) if sympy.gcd(i, phi) == 1])
    d = sympy.mod_inverse(e, phi)
    return (n, e), (n, d) # (n, e) is private key; (n, d) is public key

def q9():
    p = sympy.randprime(2**10, 2**11) # wrong
    q = sympy.randprime(2**10, 2**11) # wrong
    n = p * q
    phi = (p-1) * (q-1)
    e = e = random.choice([i for i in range(phi+1, phi*2) if sympy.gcd(i, phi) == 1]) # wrong
    d = sympy.mod_inverse(e, phi)
    return (n, e), (n, d) # (n, e) is private key; (n, d) is public key

# Question 10
# Digital signatures ensure the authenticity and integrity of a message by allowing the sender to sign a message with their private key. The recipient can then use the sender's public key to verify the signature. This process ensures that the message has not been altered and comes from the specified sender. Examine the following pseudocode for generating and verifying a digital signature.

def originalQ10():
    def generate_signature(message, privateKey):
        signature = privateKey + message 
        return signature
    def verify_signature(message, signature, publicKey):
        if publicKey in signature and message in signature:
            return True
        else:
            return False
    privateKey = 'privateKey_'
    publicKey = 'publicKey_'
    message = "Hello, world!"
    signature = generate_signature(message, privateKey)
    isValid = verify_signature(message, signature, publicKey)
    return isValid

def q10():
    def generate_signature(message, privateKey):
        signature = privateKey + message # wrong
        return signature
    def verify_signature(message, signature, publicKey):
        if publicKey in signature and message in signature: # wrong
            return True
        else:
            return False
    privateKey = 'privateKey_'
    publicKey = 'publicKey_'
    message = "Hello, world!"
    signature = generate_signature(message, privateKey)
    isValid = verify_signature(message, signature, publicKey)
    return isValid


