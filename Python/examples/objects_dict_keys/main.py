class Person:
  def __init__(self, name):
    self.name = name

people = {}
p1 = Person("Marcus")
people[p1] = True

p2 = Person("Shawn")
people[p1] = True

p3 = p1; del p1
print(p3 in people)
