package pp2;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.io.*;
import java.util.List;
import Jama.Matrix;

public class Task4 {
	public static void task4 () throws IOException {
		System.out.println("-------------- Task 4 -------------");
		String dataPath = "data/";
	
		String train_100_10 = dataPath + "train-100-10.csv";
		String trainR_100_10 = dataPath + "trainR-100-10.csv";
		String test_100_10 = dataPath + "test-100-10.csv";
		String testR_100_10 = dataPath + "testR-100-10.csv";
		double[][] train_100_10_phi = BayesianLinearRegression.readData(train_100_10);
		double[] train_100_10_t = BayesianLinearRegression.readLabels(trainR_100_10);
		double[][] test_100_10_phi = BayesianLinearRegression.readData(test_100_10);
		double[] test_100_10_t = BayesianLinearRegression.readLabels(testR_100_10);
		double[] mn_test_100_10 = BayesianLinearRegression.calculateMn(train_100_10_phi, train_100_10_t);
		double mse_test_100_10 = BayesianLinearRegression.mse(test_100_10_phi, mn_test_100_10, test_100_10_t);
		System.out.println("100-10 mse " + mse_test_100_10);
		
		String train_100_100 = dataPath + "train-100-100.csv";
		String trainR_100_100 = dataPath + "trainR-100-100.csv";
		String test_100_100 = dataPath + "test-100-100.csv";
		String testR_100_100 = dataPath + "testR-100-100.csv";
		double[][] train_100_100_phi = BayesianLinearRegression.readData(train_100_100);
		double[] train_100_100_t = BayesianLinearRegression.readLabels(trainR_100_100);
		double[][] test_100_100_phi = BayesianLinearRegression.readData(test_100_100);
		double[] test_100_100_t = BayesianLinearRegression.readLabels(testR_100_100);
		double[] mn_test_100_100 = BayesianLinearRegression.calculateMn(train_100_100_phi, train_100_100_t);
		double mse_test_100_100 = BayesianLinearRegression.mse(test_100_100_phi, mn_test_100_100, test_100_100_t);
		System.out.println("100-100 mse " + mse_test_100_100);
		
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
		double[] mn_test_50_1000_100 = BayesianLinearRegression.calculateMn(train_50_1000_100_phi, train_50_1000_100_t);
		double mse_test_50_1000_100 = BayesianLinearRegression.mse(test_1000_100_phi, mn_test_50_1000_100, test_1000_100_t);
		System.out.println("50-1000-100 mse " + mse_test_50_1000_100);
		
		double[] mn_test_100_1000_100 = BayesianLinearRegression.calculateMn(train_100_1000_100_phi, train_100_1000_100_t);
		double mse_test_100_1000_100 = BayesianLinearRegression.mse(test_1000_100_phi, mn_test_100_1000_100, test_1000_100_t);
		System.out.println("100-1000-100 mse " + mse_test_100_1000_100);
		
		double[] mn_test_150_1000_100 = BayesianLinearRegression.calculateMn(train_150_1000_100_phi, train_150_1000_100_t);
		double mse_test_150_1000_100 = BayesianLinearRegression.mse(test_1000_100_phi, mn_test_150_1000_100, test_1000_100_t);;
		System.out.println("150-1000-100 mse " + mse_test_150_1000_100);
		
		double[] mn_test_1000_100 = BayesianLinearRegression.calculateMn(train_1000_100_phi, train_1000_100_t);
		double mse_test_1000_100 = BayesianLinearRegression.mse(test_1000_100_phi, mn_test_1000_100, test_1000_100_t);
		System.out.println("1000-100 mse " + mse_test_1000_100);
		
		String train_crime = dataPath + "train-crime.csv";
		String trainR_crime = dataPath + "trainR-crime.csv";
		String test_crime = dataPath + "test-crime.csv";
		String testR_crime = dataPath + "testR-crime.csv";
		double[][] train_crime_phi = BayesianLinearRegression.readData(train_crime);
		double[] train_crime_t = BayesianLinearRegression.readLabels(trainR_crime);
		double[][] test_crime_phi = BayesianLinearRegression.readData(test_crime);
		double[] test_crime_t = BayesianLinearRegression.readLabels(testR_crime);
		double[] mn_test_crime = BayesianLinearRegression.calculateMn(train_crime_phi, train_crime_t);
		double mse_test_crime = BayesianLinearRegression.mse(test_crime_phi, mn_test_crime, test_crime_t);
		System.out.println("crime mse " + mse_test_crime);
		
		String train_wine = dataPath + "train-wine.csv";
		String trainR_wine = dataPath + "trainR-wine.csv";
		String test_wine = dataPath + "test-wine.csv";
		String testR_wine = dataPath + "testR-wine.csv";
		double[][] train_wine_phi = BayesianLinearRegression.readData(train_wine);
		double[] train_wine_t = BayesianLinearRegression.readLabels(trainR_wine);
		double[][] test_wine_phi = BayesianLinearRegression.readData(test_wine);
		double[] test_wine_t = BayesianLinearRegression.readLabels(testR_wine);
		double[] mn_test_wine = BayesianLinearRegression.calculateMn(train_wine_phi, train_wine_t);
		double mse_test_wine = BayesianLinearRegression.mse(test_wine_phi, mn_test_wine, test_wine_t);
		System.out.println("wine mse " + mse_test_wine);
	}
}
