# Python Fundamentals for Machine Learning

## 1. Data Types and Variables

### Basic Data Types
```python
# Integers
age = 25
count = -10
binary = 0b1010      # Binary
octal = 0o17        # Octal
hexadecimal = 0xFF  # Hexadecimal

# Floats
height = 5.9
pi = 3.14159
scientific = 1.5e-4

# Strings
name = "Alice"
multiline = """Multi
line
string"""

# Boolean
is_student = True
has_passed = False

# None
result = None
```

### Type Checking and Conversion
```python
type(42)           # <class 'int'>
type(3.14)         # <class 'float'>
type("hello")      # <class 'str'>
type([1, 2, 3])    # <class 'list'>

# Type conversion
int("42")          # 42
float("3.14")      # 3.14
str(123)           # "123"
bool(1)            # True
bool(0)            # False
```

## 2. Data Structures

### Lists - Ordered, Mutable, Indexed
```python
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2.5, "three", True]  # Mixed types

# Indexing
fruits[0]          # "apple"
fruits[-1]         # "cherry" (last element)

# Slicing
fruits[1:3]        # ["banana", "cherry"]
fruits[:2]         # ["apple", "banana"]
fruits[::2]        # ["apple", "cherry"] (every 2nd)
fruits[::-1]       # ["cherry", "banana", "apple"] (reversed)

# Methods
fruits.append("date")
fruits.remove("apple")
fruits.pop(0)      # Remove and return first
fruits.insert(1, "blueberry")
fruits.sort()
fruits.reverse()

# List comprehension
squares = [x**2 for x in range(5)]        # [0, 1, 4, 9, 16]
even = [x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]
```

### Tuples - Ordered, Immutable
```python
coords = (10, 20, 30)  # Fixed size
coords[0]              # 10
# coords[0] = 5        # Error! Immutable

# Unpacking
x, y, z = coords       # x=10, y=20, z=30

# Advantages: faster, hashable (can use as dict keys)
```

### Dictionaries - Key-Value Pairs, Mutable
```python
student = {
    "name": "Alice",
    "age": 20,
    "gpa": 3.8,
    "courses": ["Python", "ML", "AI"]
}

# Access
student["name"]                # "Alice"
student.get("email", "N/A")   # "N/A" (default if key doesn't exist)

# Modification
student["age"] = 21
student["email"] = "alice@example.com"

# Methods
student.keys()     # dict_keys(['name', 'age', ...])
student.values()   # dict_values(['Alice', 21, ...])
student.items()    # [('name', 'Alice'), ('age', 21), ...]

# Iteration
for key, value in student.items():
    print(f"{key}: {value}")
```

### Sets - Unordered, Unique, Mutable
```python
colors = {"red", "green", "blue", "red"}  # {"red", "green", "blue"}
unique_nums = set([1, 1, 2, 2, 3, 3])     # {1, 2, 3}

# Operations
colors.add("yellow")
colors.remove("red")

# Set operations
set1 = {1, 2, 3}
set2 = {2, 3, 4}
set1 & set2        # {2, 3} (intersection)
set1 | set2        # {1, 2, 3, 4} (union)
set1 - set2        # {1} (difference)
```

## 3. Control Flow

### Conditional Statements
```python
age = 25

if age < 13:
    print("Child")
elif age < 18:
    print("Teenager")
else:
    print("Adult")

# Ternary operator
status = "Adult" if age >= 18 else "Minor"

# Multiple conditions
if age > 18 and income > 30000:
    print("Can apply for loan")

if country == "USA" or country == "Canada":
    print("North America")

if not is_member:
    print("Please register")
```

### Loops
```python
# While loop
count = 0
while count < 5:
    print(count)
    count += 1

# For loop
for i in range(5):              # 0, 1, 2, 3, 4
    print(i)

for i in range(1, 6):           # 1, 2, 3, 4, 5
    print(i)

for i in range(0, 10, 2):       # 0, 2, 4, 6, 8
    print(i)

# Iterating over lists
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# Enumerate (get index and value)
for idx, fruit in enumerate(fruits):
    print(f"{idx}: {fruit}")

# Dictionary iteration
student = {"name": "Alice", "age": 20}
for key, value in student.items():
    print(f"{key}: {value}")

# Break and continue
for i in range(10):
    if i == 3:
        continue    # Skip to next iteration
    if i == 7:
        break       # Exit loop
    print(i)
```

## 4. Functions

### Function Basics
```python
def greet(name):
    """This function greets someone."""
    return f"Hello, {name}!"

greet("Alice")  # "Hello, Alice!"

# Default arguments
def add(a, b=5):
    return a + b

add(10)         # 15
add(10, 3)      # 13

# Variable-length arguments
def sum_all(*args):    # *args as tuple
    return sum(args)

sum_all(1, 2, 3, 4, 5)  # 15

# Keyword arguments
def print_info(name, age, city="Unknown"):
    print(f"{name}, {age}, {city}")

print_info(name="Alice", age=20, city="NYC")

# **kwargs (keyword arguments as dict)
def print_kwargs(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_kwargs(name="Alice", age=20, city="NYC")
```

### Lambda Functions
```python
# Anonymous function
square = lambda x: x ** 2
square(5)           # 25

# With multiple arguments
add = lambda x, y: x + y
add(3, 4)           # 7

# With map
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))  # [1, 4, 9, 16, 25]

# With filter
even_nums = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4]

# With sorted
students = [("Alice", 25), ("Bob", 20), ("Charlie", 23)]
sorted_by_age = sorted(students, key=lambda x: x[1])
```

## 5. String Operations

```python
text = "Hello World"

# Slicing
text[0:5]           # "Hello"
text[::-1]          # "dlroW olleH" (reversed)

# Methods
text.lower()        # "hello world"
text.upper()        # "HELLO WORLD"
text.capitalize()   # "Hello world"
text.title()        # "Hello World"
text.strip()        # Remove leading/trailing whitespace
text.split()        # ["Hello", "World"]
text.replace("World", "Python")  # "Hello Python"

# String formatting
name = "Alice"
age = 25
f"My name is {name} and I'm {age}"  # f-string (Python 3.6+)
f"Pi is approximately {3.14159:.2f}"  # "Pi is approximately 3.14"

"{} is {} years old".format(name, age)
"%s is %d years old" % (name, age)
```

## 6. File Handling

```python
# Writing to file
with open("data.txt", "w") as file:
    file.write("Hello, World!\n")
    file.write("This is line 2\n")

# Reading from file
with open("data.txt", "r") as file:
    content = file.read()           # Read entire file
    file.seek(0)                    # Reset to beginning
    lines = file.readlines()        # Read as list of lines

# Appending to file
with open("data.txt", "a") as file:
    file.write("\nAppended line\n")

# Working with CSV
import csv
with open("data.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row)
```

## 7. Exception Handling

```python
try:
    num = int("abc")          # ValueError
    result = 10 / 0           # ZeroDivisionError
except ValueError:
    print("Invalid integer")
except ZeroDivisionError:
    print("Cannot divide by zero")
except Exception as e:
    print(f"Error: {e}")
else:
    print("No error occurred")  # Executes if no exception
finally:
    print("Cleanup code")      # Always executes
```

## 8. Classes and Objects

```python
class Student:
    def __init__(self, name, age, gpa):
        self.name = name
        self.age = age
        self.gpa = gpa
    
    def display_info(self):
        return f"{self.name}, {self.age}, GPA: {self.gpa}"
    
    def is_excellent(self):
        return self.gpa >= 3.8

# Creating objects
alice = Student("Alice", 20, 3.9)
bob = Student("Bob", 21, 3.5)

alice.display_info()      # "Alice, 20, GPA: 3.9"
alice.is_excellent()      # True

# Inheritance
class GraduateStudent(Student):
    def __init__(self, name, age, gpa, thesis_title):
        super().__init__(name, age, gpa)
        self.thesis_title = thesis_title
```

## Key Takeaways

- Lists are mutable, tuples are immutable
- Dictionaries are perfect for structured data
- List comprehensions are powerful and Pythonic
- Always use `with` statement for file handling
- Lambda functions are useful for short, one-time operations
- Exception handling makes code robust
- Classes organize code and enable reusability
