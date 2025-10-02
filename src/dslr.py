import describe
import scatter_plot
import histogram
import logreg_predict
import logreg_train
import pair_plot

def getHelp():
	print("If you want to plot a graph from the dataset, you can use the following commands :")
	print("\t-histogram : To get the visual results of the different students from each houses.")
	print("\t-scatter_plot : To visualy compare the results of the houses in two differents classes.")
	print("\t-pair_plot : To get an overall view of all possible graphs.")
	print("And then you can train and predict using the model with these commands:")
	print("\t-predict <dataset path> [weights.csv]")
	print("\t-train <dataset path> ")
	print("And you can display this message again with:")
	print("\t-help")
	print("Finally, to exit the program, type:")
	print("\t-exit")

if __name__ == "__main__":
	print("Welcome to DSLR!")
	getHelp()
	Running = True
	while Running:
		try:
			cmd = input("sorting hat> ").strip().split()

			if not cmd:
				continue  # empty input, skip

			if cmd[0] == "help":
				getHelp()

			elif cmd[0] == "histogram":
				histogram.mainHistogram()

			elif cmd[0] == "scatter_plot":
				scatter_plot.main()

			elif cmd[0] == "pair_plot":
				pair_plot.main()

			elif cmd[0] == "train":
				if len(cmd) > 1:
					dataset_path = cmd[1]
					logreg_train.main(["name", dataset_path])
				else:
					print("❌ Missing dataset path. Usage: train [dataset path]")

			elif cmd[0] == "predict":
				if len(cmd) > 1:
					dataset_path = []
					for key in cmd:
						dataset_path.append(key)
					logreg_predict.main(dataset_path)
				else:
					print("❌ Missing dataset path. Usage: predict [dataset path]")

			elif cmd[0] in ("exit", "quit"):
				print("Goodbye 👋")
				Running = False

			else:
				print(f"❌ Unknown command: {cmd[0]} (type 'help' to see available commands)")

		except SystemExit as e:
			print(f"⚠️ Function tried to exit with code {e.code}, but CLI continues.")


