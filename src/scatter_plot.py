import pandas as pd
import matplotlib.pyplot as plt

def getCourseContent(course_a, course_b):
	df = pd.read_csv("dataset/dataset_train.csv")

	houses = {"Gryffindor": [], "Ravenclaw": [], "Slytherin": [], "Hufflepuff": []}

	for _, row in df.iterrows():
		house = row["Hogwarts House"]
		if not pd.isna(row[course_a]) and not pd.isna(row[course_b]):
			houses[house].append([row[course_a], row[course_b]])

	result = {house: pd.DataFrame(values, columns=[course_a, course_b])
			  for house, values in houses.items()}

	return result



def plotCourseScatter(course_data, course_a, course_b):
	colors = {
		"Gryffindor": "red",
		"Ravenclaw": "blue",
		"Slytherin": "green",
		"Hufflepuff": "gold"
	}

	plt.figure(figsize=(8, 6))

	for house, df in course_data.items(): #.items() returns a list-like view of key–value pairs:
		if not df.empty:
			plt.scatter(
				df[course_a],
				df[course_b],
				label=house,
				color=colors.get(house, "black"),
				alpha=0.7,
				edgecolors="black"
			)

	plt.xlabel(course_a)
	plt.ylabel(course_b)
	plt.title(f"{course_a} vs {course_b} by House")
	plt.legend()
	plt.grid(True, linestyle="--", alpha=0.5)

	plt.show()

def ultimateScatterPlot(course_a, course_b):
	data = getCourseContent(course_a, course_b)
	plotCourseScatter(data, course_a, course_b)

if __name__ == "__main__":
	ultimateScatterPlot("Arithmancy", "Astronomy")

