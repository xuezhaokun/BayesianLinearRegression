package pp2;

import java.io.IOException;

public class Task3 {
	public static void task3 () throws IOException {
		System.out.println("-------------- Task 3 -------------");
		String dataPath = "data/";

		String train_100_10 = dataPath + "train-100-10.csv";
		String trainR_100_10 = dataPath + "trainR-100-10.csv";
		String test_100_10 = dataPath + "test-100-10.csv";
		String testR_100_10 = dataPath + "testR-100-10.csv";
		double[][] train_100_10_phi = BayesianLinearRegression.readData(train_100_10);
		double[] train_100_10_t = BayesianLinearRegression.readLabels(trainR_100_10);
		double[][] train_100_10_with_labels = BayesianLinearRegression.combineDataWithLabels(train_100_10_phi, train_100_10_t);
		double[][] test_100_10_phi = BayesianLinearRegression.readData(test_100_10);
		double[] test_100_10_t = BayesianLinearRegression.readLabels(testR_100_10);
		double train_100_10_lambda = BayesianLinearRegression.implementTenFoldCross(train_100_10_with_labels);
		double[] train_100_10_w = BayesianLinearRegression.calculateW(train_100_10_lambda, train_100_10_phi, train_100_10_t);
		double test_100_10_mse = BayesianLinearRegression.mse(test_100_10_phi, train_100_10_w, test_100_10_t);
		System.out.println("lambda: " + train_100_10_lambda + " test-100-10 mse: " + test_100_10_mse);
		
		String train_100_100 = dataPath + "train-100-100.csv";
		String trainR_100_100 = dataPath + "trainR-100-100.csv";
		String test_100_100 = dataPath + "test-100-100.csv";
		String testR_100_100 = dataPath + "testR-100-100.csv";
		double[][] train_100_100_phi = BayesianLinearRegression.readData(train_100_100);
		double[] train_100_100_t = BayesianLinearRegression.readLabels(trainR_100_100);
		double[][] train_100_100_with_labels = BayesianLinearRegression.combineDataWithLabels(train_100_100_phi, train_100_100_t);
		double[][] test_100_100_phi = BayesianLinearRegression.readData(test_100_100);
		double[] test_100_100_t = BayesianLinearRegression.readLabels(testR_100_100);
		double train_100_100_lambda = BayesianLinearRegression.implementTenFoldCross(train_100_100_with_labels);
		double[] train_100_100_w = BayesianLinearRegression.calculateW(train_100_100_lambda, train_100_100_phi, train_100_100_t);
		double test_100_100_mse = BayesianLinearRegression.mse(test_100_100_phi, train_100_100_w, test_100_100_t);
		System.out.println("lambda: " + train_100_100_lambda + " test-100-100 mse: " + test_100_100_mse);
		
		String train_50_1000_100 = dataPath + "train-(50)1000-100.csv";
		String train_100_1000_100 = dataPath + "train-(100)1000-100.csv";
		String train_150_1000_100 = dataPath + "train-(150)1000-100.csv";
		String train_1000_100 = dataPath + "train-1000-100.csv";
		String trainR_1000_100 = dataPath + "trainR-1000-100.csv";
		String test_1000_100 = dataPath + "test-1000-100.csv";
		String testR_1000_100 = dataPath + "testR-1000-100.csv";
		
		double[][] train_50_1000_100_phi = BayesianLinearRegression.readData(train_50_1000_100);
		double[][] train_100_1000_100_phi = BayesianLinearRegression.readData(train_100_1000_100);
		double[][] train_150_1000_100_phi = BayesianLinearRegression.readData(train_150_1000_100);
		double[][] train_1000_100_phi = BayesianLinearRegression.readData(train_1000_100);
		double[] train_1000_100_t = BayesianLinearRegression.readLabels(trainR_1000_100);
		double[] train_50_1000_100_t = BayesianLinearRegression.getFirstNLabels(train_1000_100_t, 50);
		double[] train_100_1000_100_t = BayesianLinearRegression.getFirstNLabels(train_1000_100_t, 100);
		double[] train_150_1000_100_t = BayesianLinearRegression.getFirstNLabels(train_1000_100_t, 150);
		double[][] train_50_1000_100_with_labels = BayesianLinearRegression.combineDataWithLabels(train_50_1000_100_phi, train_50_1000_100_t);
		double[][] train_100_1000_100_with_labels = BayesianLinearRegression.combineDataWithLabels(train_100_1000_100_phi, train_100_1000_100_t);
		double[][] train_150_1000_100_with_labels = BayesianLinearRegression.combineDataWithLabels(train_150_1000_100_phi, train_150_1000_100_t);
		double[][] train_1000_100_with_labels = BayesianLinearRegression.combineDataWithLabels(train_1000_100_phi, train_1000_100_t);
		
		double[][] test_1000_100_phi = BayesianLinearRegression.readData(test_1000_100);
		double[] test_1000_100_t = BayesianLinearRegression.readLabels(testR_1000_100);
		
		double train_50_1000_100_lambda = BayesianLinearRegression.implementTenFoldCross(train_50_1000_100_with_labels);
		double[] train_50_1000_100_w = BayesianLinearRegression.calculateW(train_50_1000_100_lambda, train_50_1000_100_phi, train_50_1000_100_t);
		double test_50_1000_100_mse = BayesianLinearRegression.mse(test_1000_100_phi, train_50_1000_100_w, test_1000_100_t);
		System.out.println("lambda: " + train_50_1000_100_lambda + " test-50-1000-100 mse: " + test_50_1000_100_mse);
		
		double train_100_1000_100_lambda = BayesianLinearRegression.implementTenFoldCross(train_100_1000_100_with_labels);
		double[] train_100_1000_100_w = BayesianLinearRegression.calculateW(train_100_1000_100_lambda, train_100_1000_100_phi, train_100_1000_100_t);
		double test_100_1000_100_mse = BayesianLinearRegression.mse(test_1000_100_phi, train_100_1000_100_w, test_1000_100_t);
		System.out.println("lambda: " + train_100_1000_100_lambda + " test-100-1000-100 mse: " + test_100_1000_100_mse);
		
		double train_150_1000_100_lambda = BayesianLinearRegression.implementTenFoldCross(train_150_1000_100_with_labels);
		double[] train_150_1000_100_w = BayesianLinearRegression.calculateW(train_150_1000_100_lambda, train_150_1000_100_phi, train_150_1000_100_t);
		double test_150_1000_100_mse = BayesianLinearRegression.mse(test_1000_100_phi, train_150_1000_100_w, test_1000_100_t);
		System.out.println("lambda: " + train_150_1000_100_lambda + " test-150-1000-100 mse: " + test_150_1000_100_mse);
		
		double train_1000_100_lambda = BayesianLinearRegression.implementTenFoldCross(train_1000_100_with_labels);
		double[] train_1000_100_w = BayesianLinearRegression.calculateW(train_1000_100_lambda, train_1000_100_phi, train_1000_100_t);
		double test_1000_100_mse = BayesianLinearRegression.mse(test_1000_100_phi, train_1000_100_w, test_1000_100_t);
		System.out.println("lambda: " + train_1000_100_lambda + " test-1000-100 mse: " + test_1000_100_mse);
		
		String train_crime = dataPath + "train-crime.csv";
		String trainR_crime = dataPath + "trainR-crime.csv";
		String test_crime = dataPath + "test-crime.csv";
		String testR_crime = dataPath + "testR-crime.csv";
		double[][] train_crime_phi = BayesianLinearRegression.readData(train_crime);
		double[] train_crime_t = BayesianLinearRegression.readLabels(trainR_crime);
		double[][] train_crime_with_labels = BayesianLinearRegression.combineDataWithLabels(train_crime_phi, train_crime_t);
		
		double[][] test_crime_phi = BayesianLinearRegression.readData(test_crime);
		double[] test_crime_t = BayesianLinearRegression.readLabels(testR_crime);
		
		double train_crime_lambda = BayesianLinearRegression.implementTenFoldCross(train_crime_with_labels);
		double[] train_crime_w = BayesianLinearRegression.calculateW(train_crime_lambda, train_crime_phi, train_crime_t);
		double test_crime_mse = BayesianLinearRegression.mse(test_crime_phi, train_crime_w, test_crime_t);
		System.out.println("lambda: " + train_crime_lambda + " test-crime mse: " + test_crime_mse);
		
		
		String train_wine = dataPath + "train-wine.csv";
		String trainR_wine = dataPath + "trainR-wine.csv";
		String test_wine = dataPath + "test-wine.csv";
		String testR_wine = dataPath + "testR-wine.csv";
		double[][] train_wine_phi = BayesianLinearRegression.readData(train_wine);
		double[] train_wine_t = BayesianLinearRegression.readLabels(trainR_wine);
		double[][] train_wine_with_labels = BayesianLinearRegression.combineDataWithLabels(train_wine_phi, train_wine_t);
		
		double[][] test_wine_phi = BayesianLinearRegression.readData(test_wine);
		double[] test_wine_t = BayesianLinearRegression.readLabels(testR_wine);
		
		double train_wine_lambda = BayesianLinearRegression.implementTenFoldCross(train_wine_with_labels);
		double[] train_wine_w = BayesianLinearRegression.calculateW(train_wine_lambda, train_wine_phi, train_wine_t);
		double test_wine_mse = BayesianLinearRegression.mse(test_wine_phi, train_wine_w, test_wine_t);
		System.out.println("lambda: " + train_wine_lambda + " test-wine mse: " + test_wine_mse);
	}
}
