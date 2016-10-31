package pp2;

import java.io.IOException;
import java.io.PrintWriter;

public class Task1 {
	
	public static void task1 () throws IOException {
		System.out.println("-------------- Task 1 -------------");
		String dataPath = "data/";

		String train_100_10 = dataPath + "train-100-10.csv";
		String trainR_100_10 = dataPath + "trainR-100-10.csv";
		String test_100_10 = dataPath + "test-100-10.csv";
		String testR_100_10 = dataPath + "testR-100-10.csv";
		double[][] train_100_10_phi = BayesianLinearRegression.readData(train_100_10);
		double[] train_100_10_t = BayesianLinearRegression.readLabels(trainR_100_10);
		double[][] test_100_10_phi = BayesianLinearRegression.readData(test_100_10);
		double[] test_100_10_t = BayesianLinearRegression.readLabels(testR_100_10);
		double[] mse_train_100_10 = new double[151];
		double[] mse_test_100_10 = new double[151];
		PrintWriter writer_mse_100_10 = new PrintWriter("results/mse_100_10", "UTF-8");

		
		String train_100_100 = dataPath + "train-100-100.csv";
		String trainR_100_100 = dataPath + "trainR-100-100.csv";
		String test_100_100 = dataPath + "test-100-100.csv";
		String testR_100_100 = dataPath + "testR-100-100.csv";
		double[][] train_100_100_phi = BayesianLinearRegression.readData(train_100_100);
		double[] train_100_100_t = BayesianLinearRegression.readLabels(trainR_100_100);
		double[][] test_100_100_phi = BayesianLinearRegression.readData(test_100_100);
		double[] test_100_100_t = BayesianLinearRegression.readLabels(testR_100_100);
		double[] mse_train_100_100 = new double[151];
		double[] mse_test_100_100 = new double[151];
		PrintWriter writer_mse_100_100 = new PrintWriter("results/mse_100_100", "UTF-8");
		
		
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
		double[][] test_1000_100_phi = BayesianLinearRegression.readData(test_1000_100);
		double[] test_1000_100_t = BayesianLinearRegression.readLabels(testR_1000_100);
		
		double[] mse_train_50_1000_100 = new double[151];
		double[] mse_train_100_1000_100 = new double[151];
		double[] mse_train_150_1000_100 = new double[151];
		double[] mse_train_1000_100 = new double[151];
		
		double[] mse_test_50_1000_100 = new double[151];
		double[] mse_test_100_1000_100 = new double[151];
		double[] mse_test_150_1000_100 = new double[151];
		double[] mse_test_1000_100 = new double[151];
		PrintWriter writer_mse_50_1000_100 = new PrintWriter("results/mse_50_1000_100", "UTF-8");
		PrintWriter writer_mse_100_1000_100  = new PrintWriter("results/mse_100_1000_100", "UTF-8");
		PrintWriter writer_mse_150_1000_100  = new PrintWriter("results/mse_150_1000_100", "UTF-8");
		PrintWriter writer_mse_1000_100  = new PrintWriter("results/mse_1000_100", "UTF-8");
		
		String train_crime = dataPath + "train-crime.csv";
		String trainR_crime = dataPath + "trainR-crime.csv";
		String test_crime = dataPath + "test-crime.csv";
		String testR_crime = dataPath + "testR-crime.csv";
		double[][] train_crime_phi = BayesianLinearRegression.readData(train_crime);
		double[] train_crime_t = BayesianLinearRegression.readLabels(trainR_crime);
		double[][] test_crime_phi = BayesianLinearRegression.readData(test_crime);
		double[] test_crime_t = BayesianLinearRegression.readLabels(testR_crime);
		double[] mse_train_crime = new double[151];
		double[] mse_test_crime = new double[151];
		PrintWriter writer_mse_crime = new PrintWriter("results/mse_crime", "UTF-8");
		
		String train_wine = dataPath + "train-wine.csv";
		String trainR_wine = dataPath + "trainR-wine.csv";
		String test_wine = dataPath + "test-wine.csv";
		String testR_wine = dataPath + "testR-wine.csv";
		double[][] train_wine_phi = BayesianLinearRegression.readData(train_wine);
		double[] train_wine_t = BayesianLinearRegression.readLabels(trainR_wine);
		double[][] test_wine_phi = BayesianLinearRegression.readData(test_wine);
		double[] test_wine_t = BayesianLinearRegression.readLabels(testR_wine);
		double[] mse_train_wine = new double[151];
		double[] mse_test_wine = new double[151];
		PrintWriter writer_mse_wine = new PrintWriter("results/mse_wine", "UTF-8");
		
		for (double lambda = 0; lambda < 151; lambda++) {
			int index = (int) lambda;
			double[] w_100_10 = BayesianLinearRegression.calculateW(lambda, train_100_10_phi, train_100_10_t);
			double[] w_100_100 = BayesianLinearRegression.calculateW(lambda, train_100_100_phi, train_100_100_t);
			double[] w_50_1000_100 = BayesianLinearRegression.calculateW(lambda, train_50_1000_100_phi, train_50_1000_100_t);
			double[] w_100_1000_100 = BayesianLinearRegression.calculateW(lambda, train_100_1000_100_phi, train_100_1000_100_t);
			double[] w_150_1000_100 = BayesianLinearRegression.calculateW(lambda, train_150_1000_100_phi, train_150_1000_100_t);
			double[] w_1000_100 = BayesianLinearRegression.calculateW(lambda, train_1000_100_phi, train_1000_100_t);
			double[] w_crime = BayesianLinearRegression.calculateW(lambda, train_crime_phi, train_crime_t);
			double[] w_wine = BayesianLinearRegression.calculateW(lambda, train_wine_phi, train_wine_t);
			
			mse_train_100_10[index] = BayesianLinearRegression.mse(train_100_10_phi, w_100_10, train_100_10_t);
	        mse_test_100_10[index] = BayesianLinearRegression.mse(test_100_10_phi, w_100_10, test_100_10_t);
	        writer_mse_100_10.println(index + " " + mse_train_100_10[index] + " " + mse_test_100_10[index] + " 3.78");
	        
			mse_train_100_100[index] = BayesianLinearRegression.mse(train_100_100_phi, w_100_100, train_100_100_t);
	        mse_test_100_100[index] = BayesianLinearRegression.mse(test_100_100_phi, w_100_100, test_100_100_t);
	        writer_mse_100_100.println(index + " " + mse_train_100_100[index] + " " + mse_test_100_100[index] + " 3.78");
	        
			mse_train_50_1000_100[index] = BayesianLinearRegression.mse(train_50_1000_100_phi, w_50_1000_100, train_50_1000_100_t);
			mse_train_100_1000_100[index] = BayesianLinearRegression.mse(train_100_1000_100_phi, w_100_1000_100, train_100_1000_100_t);
			mse_train_150_1000_100[index] = BayesianLinearRegression.mse(train_150_1000_100_phi, w_150_1000_100, train_150_1000_100_t);
			mse_train_1000_100[index] = BayesianLinearRegression.mse(train_1000_100_phi, w_1000_100, train_1000_100_t);

			
			mse_test_50_1000_100[index] = BayesianLinearRegression.mse(test_1000_100_phi, w_50_1000_100, test_1000_100_t);
			mse_test_100_1000_100[index] = BayesianLinearRegression.mse(test_1000_100_phi, w_100_1000_100, test_1000_100_t);
			mse_test_150_1000_100[index] = BayesianLinearRegression.mse(test_1000_100_phi, w_150_1000_100, test_1000_100_t);
			mse_test_1000_100[index] = BayesianLinearRegression.mse(test_1000_100_phi, w_1000_100, test_1000_100_t);
			writer_mse_50_1000_100.println(index + " " + mse_train_50_1000_100[index] + " " + mse_test_50_1000_100[index] + " 4.015");
			writer_mse_100_1000_100.println(index + " " + mse_train_100_1000_100[index] + " " + mse_test_100_1000_100[index] + " 4.015");
			writer_mse_150_1000_100.println(index + " " + mse_train_150_1000_100[index] + " " + mse_test_150_1000_100[index] + " 4.015");
	        writer_mse_1000_100.println(index + " " + mse_train_1000_100[index] + " " + mse_test_1000_100[index] + " 4.015");
	        
			mse_train_crime[index] = BayesianLinearRegression.mse(train_crime_phi, w_crime, train_crime_t);
	        mse_test_crime[index] = BayesianLinearRegression.mse(test_crime_phi, w_crime, test_crime_t);
	        writer_mse_crime.println(index + " " + mse_train_crime[index]+  " " + mse_test_crime[index]);
	        
			mse_train_wine[index] = BayesianLinearRegression.mse(train_wine_phi, w_wine, train_wine_t);
	        mse_test_wine[index] = BayesianLinearRegression.mse(test_wine_phi, w_wine, test_wine_t);
	        writer_mse_wine.println(index + " " + mse_train_wine[index] + " " + mse_test_wine[index]);
		}
		writer_mse_100_10.close();
		writer_mse_100_100.close();
		writer_mse_50_1000_100.close();
		writer_mse_100_1000_100.close();
		writer_mse_150_1000_100.close();
		writer_mse_1000_100.close();
		writer_mse_crime.close();
		writer_mse_wine.close();
		
		double[] optimal_test_100_10 = BayesianLinearRegression.findMinMse(mse_test_100_10);
		System.out.println("test-100-10 lambda: " + optimal_test_100_10[0] + " mse: " + optimal_test_100_10[1]);

		double[] optimal_test_100_100 = BayesianLinearRegression.findMinMse(mse_test_100_100);
		System.out.println("test-100-100 lambda: " + optimal_test_100_100[0] + " mse: " + optimal_test_100_100[1]);
		
		double[] optimal_test_50_1000_100 = BayesianLinearRegression.findMinMse(mse_test_50_1000_100);
		System.out.println("test-50-1000-100 lambda: " + optimal_test_50_1000_100[0] + " mse: " + optimal_test_50_1000_100[1]);
		double[] optimal_test_100_1000_100 = BayesianLinearRegression.findMinMse(mse_test_100_1000_100);
		System.out.println("test-100-1000-100 lambda: " + optimal_test_100_1000_100[0] + " mse: " + optimal_test_100_1000_100[1]);
		double[] optimal_test_150_1000_100 = BayesianLinearRegression.findMinMse(mse_test_150_1000_100);
		System.out.println("test-150-1000-100 lambda: " + optimal_test_150_1000_100[0] + " mse: " + optimal_test_150_1000_100[1]);
		double[] optimal_test_1000_100 = BayesianLinearRegression.findMinMse(mse_test_1000_100);
		System.out.println("test-1000-100lambda: " + optimal_test_1000_100[0] + " mse: " + optimal_test_1000_100[1]);
		
		double[] optimal_test_crime = BayesianLinearRegression.findMinMse(mse_test_crime);
		System.out.println("test-crime lambda: " + optimal_test_crime[0] + " mse: " + optimal_test_crime[1]);
		
		double[] optimal_test_wine = BayesianLinearRegression.findMinMse(mse_test_wine);
		System.out.println("test-wine lambda: " + optimal_test_wine[0] + " mse: " + optimal_test_wine[1]);
	}
}
