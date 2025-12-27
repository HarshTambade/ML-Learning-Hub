#!/usr/bin/env python3
"""Python Basics Examples

Comprehensive examples of fundamental Python programming concepts.
"""

# ===== DATA TYPES =====
def data_types_examples():
    """Demonstrate different Python data types."""
    print("\n=== Data Types ===")
    
    # Integers
    num_int = 42
    print(f"Integer: {num_int}, type: {type(num_int)}")
    
    # Floats
    num_float = 3.14
    print(f"Float: {num_float}, type: {type(num_float)}")
    
    # Strings
    text = "Hello, Python!"
    print(f"String: {text}, type: {type(text)}")
    
    # Booleans
    is_valid = True
    print(f"Boolean: {is_valid}, type: {type(is_valid)}")
    
    # Type conversion
    str_num = "123"
    converted = int(str_num)
    print(f"Converted '{str_num}' to {converted}")

# ===== COLLECTIONS =====
def collections_examples():
    """Demonstrate lists, tuples, sets, and dictionaries."""
    print("\n=== Collections ===")
    
    # Lists (mutable, ordered)
    my_list = [1, 2, 3, 4, 5]
    print(f"List: {my_list}")
    my_list.append(6)
    print(f"After append: {my_list}")
    print(f"First element: {my_list[0]}")
    print(f"Slice [1:3]: {my_list[1:3]}")
    
    # Tuples (immutable, ordered)
    my_tuple = (10, 20, 30)
    print(f"\nTuple: {my_tuple}")
    print(f"Length: {len(my_tuple)}")
    
    # Sets (unordered, unique)
    my_set = {1, 2, 2, 3, 3, 3}
    print(f"\nSet: {my_set}")
    print(f"Set length: {len(my_set)}")
    
    # Dictionaries (key-value pairs)
    person = {'name': 'Alice', 'age': 25, 'city': 'NYC'}
    print(f"\nDictionary: {person}")
    print(f"Name: {person['name']}")
    print(f"Keys: {person.keys()}")
    print(f"Values: {person.values()}")

# ===== CONTROL FLOW =====
def control_flow_examples():
    """Demonstrate if-else, loops, and conditions."""
    print("\n=== Control Flow ===")
    
    # If-else
    age = 20
    if age < 13:
        print("Child")
    elif age < 18:
        print("Teenager")
    else:
        print("Adult")
    
    # For loop
    print("\nFor loop:")
    for i in range(5):
        print(f"  i = {i}")
    
    # For loop with list
    print("\nFor loop with list:")
    fruits = ['apple', 'banana', 'cherry']
    for fruit in fruits:
        print(f"  {fruit}")
    
    # While loop
    print("\nWhile loop:")
    count = 0
    while count < 3:
        print(f"  count = {count}")
        count += 1
    
    # List comprehension
    squares = [x**2 for x in range(5)]
    print(f"\nSquares: {squares}")
    
    # Conditional list comprehension
    evens = [x for x in range(10) if x % 2 == 0]
    print(f"Evens: {evens}")

# ===== FUNCTIONS =====
def function_examples():
    """Demonstrate function definitions and usage."""
    print("\n=== Functions ===")
    
    # Simple function
    def greet(name):
        return f"Hello, {name}!"
    
    print(greet("Alice"))
    
    # Function with default argument
    def power(base, exp=2):
        return base ** exp
    
    print(f"2^3 = {power(2, 3)}")
    print(f"5^2 = {power(5)}")
    
    # Function with *args (variable arguments)
    def sum_all(*args):
        return sum(args)
    
    print(f"Sum: {sum_all(1, 2, 3, 4, 5)}")
    
    # Function with **kwargs (keyword arguments)
    def print_info(**kwargs):
        for key, value in kwargs.items():
            print(f"  {key}: {value}")
    
    print("Info:")
    print_info(name="Bob", age=30, city="LA")
    
    # Lambda function
    square = lambda x: x ** 2
    print(f"Lambda square(4): {square(4)}")
    
    # Map function
    numbers = [1, 2, 3, 4, 5]
    squared = list(map(lambda x: x**2, numbers))
    print(f"Mapped squares: {squared}")
    
    # Filter function
    evens = list(filter(lambda x: x % 2 == 0, numbers))
    print(f"Filtered evens: {evens}")

# ===== STRING OPERATIONS =====
def string_operations():
    """Demonstrate string manipulation."""
    print("\n=== String Operations ===")
    
    text = "Hello World"
    
    # Basic operations
    print(f"Original: {text}")
    print(f"Uppercase: {text.upper()}")
    print(f"Lowercase: {text.lower()}")
    print(f"Length: {len(text)}")
    
    # String methods
    print(f"Starts with 'Hello': {text.startswith('Hello')}")
    print(f"Ends with 'World': {text.endswith('World')}")
    print(f"Contains 'o': {'o' in text}")
    
    # String slicing
    print(f"First 5 chars: {text[:5]}")
    print(f"Last 5 chars: {text[-5:]}")
    print(f"Every 2nd char: {text[::2]}")
    
    # String splitting and joining
    words = text.split()
    print(f"Split: {words}")
    joined = "-".join(words)
    print(f"Joined: {joined}")
    
    # String replacement
    replaced = text.replace("World", "Python")
    print(f"Replaced: {replaced}")
    
    # String formatting
    name = "Alice"
    age = 25
    print(f"F-string: {name} is {age} years old")
    print("Format method: {} is {} years old".format(name, age))
    print("% formatting: %s is %d years old" % (name, age))

# ===== FILE OPERATIONS =====
def file_operations_example():
    """Demonstrate reading and writing files."""
    print("\n=== File Operations ===")
    
    # Writing to file
    filename = "sample.txt"
    with open(filename, 'w') as f:
        f.write("Line 1\n")
        f.write("Line 2\n")
        f.write("Line 3\n")
    print(f"Written to {filename}")
    
    # Reading from file
    with open(filename, 'r') as f:
        content = f.read()
    print(f"File content:\n{content}")
    
    # Reading line by line
    print("Reading line by line:")
    with open(filename, 'r') as f:
        for line in f:
            print(f"  {line.strip()}")

# ===== EXCEPTION HANDLING =====
def exception_handling():
    """Demonstrate try-except blocks."""
    print("\n=== Exception Handling ===")
    
    # Try-except
    try:
        num = int("abc")
    except ValueError:
        print("ValueError: Could not convert string to integer")
    
    # Multiple exceptions
    try:
        my_list = [1, 2, 3]
        print(my_list[10])
    except IndexError:
        print("IndexError: Index out of range")
    except ValueError:
        print("ValueError occurred")
    
    # Try-except-else
    try:
        result = 10 / 2
    except ZeroDivisionError:
        print("Cannot divide by zero")
    else:
        print(f"Result: {result}")
    
    # Try-except-finally
    try:
        f = open("nonexistent.txt", "r")
    except FileNotFoundError:
        print("FileNotFoundError: File not found")
    finally:
        print("Finally block always executes")

# ===== CLASSES AND OOP =====
def classes_example():
    """Demonstrate class definition and usage."""
    print("\n=== Classes and OOP ===")
    
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age
        
        def greet(self):
            return f"Hello, I'm {self.name}"
        
        def birthday(self):
            self.age += 1
            print(f"{self.name} is now {self.age}")
    
    # Create objects
    person1 = Person("Alice", 25)
    person2 = Person("Bob", 30)
    
    print(person1.greet())
    print(person2.greet())
    
    person1.birthday()
    
    # Inheritance
    class Student(Person):
        def __init__(self, name, age, student_id):
            super().__init__(name, age)
            self.student_id = student_id
        
        def study(self, subject):
            print(f"{self.name} is studying {subject}")
    
    student = Student("Charlie", 20, "S12345")
    print(student.greet())
    student.study("Python")

# ===== MAIN EXECUTION =====
if __name__ == '__main__':
    print("Python Basics Examples")
    print("=" * 50)
    
    data_types_examples()
    collections_examples()
    control_flow_examples()
    function_examples()
    string_operations()
    file_operations_example()
    exception_handling()
    classes_example()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
