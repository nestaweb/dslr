import matplotlib.pyplot as plt
import csv
from utils import DSLR

def getCourseHistogram(course):
     RavenclawMarks = []
     SlytherinMarks = []
     GryffindorMarks = []
     HufflepuffMarks = []

     with open(DSLR.trainfilepath, mode='r') as file:    
          csvFileReader = csv.DictReader(file)
          for line in csvFileReader:
               house = line['Hogwarts House']
               mark = line[course]
               if mark:
                    mark = float(mark)
                    match house:
                         case "Ravenclaw":
                              RavenclawMarks.append(mark)
                         case "Slytherin":
                              SlytherinMarks.append(mark)
                         case "Gryffindor":
                              GryffindorMarks.append(mark)
                         case "Hufflepuff":
                              HufflepuffMarks.append(mark)

     allValue = [RavenclawMarks, SlytherinMarks, GryffindorMarks, HufflepuffMarks]

     _, ax = plt.subplots()
     for house in allValue:
          ax.hist(house, alpha=0.5)

     plt.legend(['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff'])
     plt.xlabel('Marks')
     plt.ylabel('Nb Student')
     plt.title(course)
     plt.show()

def selectMenu():
     choices = DSLR.getNumericalFeature()
     choices.insert(0, "All")
     i = 0
     for choice in choices:
          print(str(i) + ": " + choice)
          i+=1

     selectedChoice = int(input("Please enter your choice : "))
     if (selectedChoice > 0):
          getCourseHistogram(choices[selectedChoice])
     else: 
          for choice in choices:
               if choice != "All":
                    getCourseHistogram(choice)

def mainHistogram():
     course = selectMenu()
     courses = DSLR.getNumericalFeature()
     if (course != "All"):
          getCourseHistogram(course)
     else: 
          for course in courses:
               getCourseHistogram(course)


mainHistogram()