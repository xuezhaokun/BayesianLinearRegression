package pp2;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collections;

public class Task2 {
	public static void task2() throws IOException{
		String dataPath = "data/";
		String train_1000_100 = dataPath + "train-1000-100.csv";
		String trainR_1000_100 = dataPath + "trainR-1000-100.csv";
		String test_1000_100 = dataPath + "test-1000-100.csv";
		String testR_1000_100 = dataPath + "testR-1000-100.csv";
		PrintWriter writer_mse_learning_curve_0_1000_100  = new PrintWriter("results/learning_curve_0_1000_100 ", "UTF-8");
		PrintWriter writer_mse_learning_curve_90_1000_100  = new PrintWriter("results/learning_curve_90_1000_100 ", "UTF-8");
		PrintWriter writer_mse_learning_curve_150_1000_100  = new PrintWriter("results/learning_curve_150_1000_100 ", "UTF-8");
		
		double[][] train_1000_100_phi = BayesianLinearRegression.readData(train_1000_100);
		double[] train_1000_100_t = BayesianLinearRegression.readLabels(trainR_1000_100);
		double[][] test_1000_100_phi = BayesianLinearRegression.readData(test_1000_100);
		double[] test_1000_100_t = BayesianLinearRegression.readLabels(testR_1000_100);		
		
		double[][] data_with_labels = BayesianLinearRegression.combineDataWithLabels(train_1000_100_phi, train_1000_100_t);
		
		double[] lambdas = new double[] {0, 90, 150};
		for (double lambda : lambdas) {
			for (int n = 10; n < 810; n = n + 10) {
				double mse = 0;
				for (int m = 0; m < 10; m++) {
					Collections.shuffle(Arrays.asList(data_with_labels));
					double[][] training_data_with_labels = BayesianLinearRegression.getFirstNData(data_with_labels, n);
					double[][] trainging_data = BayesianLinearRegression.getDataFromDataWithLabels(training_data_with_labels);
					if (BayesianLinearRegression.determineSingular(lambda, trainging_data)){
						m--;
						continue;
					}
					double[] training_data_labels = BayesianLinearRegression.getLabelsFromDataWithLabels(training_data_with_labels);
					double[] w = BayesianLinearRegression.calculateW(lambda, trainging_data, training_data_labels);
					mse += BayesianLinearRegression.mse(test_1000_100_phi, w, test_1000_100_t);
				}
				double avg_mse = mse/(double)10;
				if ((int)lambda == 0) {
					writer_mse_learning_curve_0_1000_100.println(n + " " + avg_mse);
				} else if ((int)lambda == 90) {
					writer_mse_learning_curve_90_1000_100.println(n + " " + avg_mse);
				} else {
					writer_mse_learning_curve_150_1000_100.println(n + " " + avg_mse);
				}
			}
		}
		writer_mse_learning_curve_0_1000_100.close();
		writer_mse_learning_curve_90_1000_100.close();
		writer_mse_learning_curve_150_1000_100.close();
	}
}
