from day1_functions import fibonacci,is_prime,is_palindrome,char_frequency,binary_search
import unittest

class MyTest(unittest.TestCase):
    def test_fibfibonacci(self):
        self.assertEqual(fibonacci(5),[0,1,1,2,3])
        self.assertEqual(fibonacci(1),[0])
        self.assertEqual(fibonacci(0),[])

    def test_is_prime(self):
        self.assertEqual(is_prime(2),True)
        self.assertEqual(is_prime(7),True)
        self.assertEqual(is_prime(8),False)

    def test_is_palindorme(self):
        self.assertEqual(is_palindrome('abccba'),True)
        self.assertEqual(is_palindrome('acvui'),False)

    def test_char_frequency(self):
        self.assertEqual(char_frequency('lin'),{'l':1,'i':1,'n':1})

    def test_binary_search(self):
        self.assertEqual(binary_search([1,2,3,4,5,6],3),2)

if __name__ == '__main__':
    unittest.main()