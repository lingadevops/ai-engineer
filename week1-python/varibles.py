course = "Python Programming"
message = """Welcome to the Python Programming course!
This course will cover the basics of Python programming, including variables, data types, control structures"""
print(len(course))
print(len(message))
print(course[0])
print(course[-1])

first_name = "John"
last_name = "Doe"
full_name = first_name + " " + last_name
print(full_name)
full_name = f"{first_name} {last_name}"
print(full_name)
print(course.upper())
print(course.strip())

x = input("X: ")
y = int(x) + 1
print(f"X: {x}, Y: {y}")