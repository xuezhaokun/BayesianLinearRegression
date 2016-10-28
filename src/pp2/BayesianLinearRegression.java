package pp2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.io.*;
import java.util.List;
import Jama.Matrix;

public class BayesianLinearRegression {
	/**
	 * The function parses an input file to a list of strings
	 * @param filename input file name
	 * @return a matrix
	 * @throws IOException input/output exception
	 */
	public static double[][] readData(String filename) throws IOException {

        List<double[]> examples = new ArrayList<double[]>();
        double[][] phi = null;
		// try to open and read the file
		try {
			BufferedReader br = new BufferedReader(new FileReader(filename));
            String fileRead = br.readLine();
            // parse file content by space and add each token to the list
            while (fileRead != null) {
                String[] tokens = fileRead.split(",");
                double[] example = new double[tokens.length];
                for (int i = 0; i < tokens.length; i++) {
                		example[i] = Double.parseDouble(tokens[i]);
                }
                examples.add(example);
                fileRead = br.readLine();
            }
            phi = new double[examples.size()][0];
            br.close();
            
        } catch (FileNotFoundException fnfe) {
            System.err.println("file not found");
        }
		return examples.toArray(phi);
	}
	
	
	public static double[] readLabels(String filename) throws IOException {

        List<Double> labels = new ArrayList<Double>();
        double[] t = null;
		// try to open and read the file
		try {
			BufferedReader br = new BufferedReader(new FileReader(filename));
            String fileRead = br.readLine();
            // parse file content by space and add each token to the list
            while (fileRead != null) {
            		labels.add(Double.parseDouble(fileRead));
                fileRead = br.readLine();
            }
            t = new double[labels.size()];
            for (int i = 0; i < t.length; i++) {
                t[i] = labels.get(i);                // java 1.5+ style (outboxing)
             }
            br.close();
            
        } catch (FileNotFoundException fnfe) {
            System.err.println("file not found");
        }
		return t;
	}
	
	public static double[] getFirstNLabels(double[] labels, int n) {
		double[] result = new double[n];
		for (int i = 0; i < n; i++) {
			result[i] = labels[i];
		}
		return result;
	}
	
	public static double[][] getFirstNData(double[][] data, int n) {
		int dimention = data[0].length;
		double[][] results = new double[n][dimention];
		for (int i = 0; i < n; i++){
			results[i] = data[i];
		}
		return results;
	}

	public static double[][] combineDataWithLabels(double[][] data, double[] labels) {
		int dimension = data[0].length + 1;
		double[][] data_with_labels = new double[data.length][dimension];
		for (int i = 0; i < data.length; i++) {
			double[] data_label = new double [dimension];
			for (int j = 0; j < data[i].length; j++) {
				data_label[j] = data[i][j];
			}
			data_label[dimension - 1] = labels[i];
			data_with_labels[i] = data_label;
		}
		return data_with_labels;
	}
	
	public static double[][] getDataFromDataWithLabels(double[][] data_with_labels) {
		int dimension = data_with_labels[0].length - 1;
		double[][] pure_data = new double[data_with_labels.length][dimension];
		
		for (int i = 0; i < data_with_labels.length; i++) {
			for (int j = 0; j < dimension; j++) {
				pure_data[i][j] = data_with_labels[i][j];
			}
		}
		return pure_data;
	}
	
	public static double[] getLabelsFromDataWithLabels(double[][] data_with_labels) {
		double[] labels = new double[data_with_labels.length];
		int dimension = data_with_labels[0].length;
		for (int i = 0; i < data_with_labels.length; i++) {
			labels[i] = data_with_labels[i][dimension-1];
		}
		return labels;
	}
	
	public static double implementTenFoldCross(double[][] data_with_labels) {
		double[] lambdas_mse = new double[151];
		int data_size = data_with_labels.length;
//		int fold_size = data_size/10;
//		int dimension = data_with_labels[0].length;
		HashMap<Integer, double[][]> folds = BayesianLinearRegression.splitFolds(data_with_labels);
		//System.out.println(folds.toString());
		for (double lambda = 0; lambda < 151; lambda++) {
//			double[][] train_data_with_labels = new double[data_size-fold_size][dimension];
//			double[][] test_data_with_labels = new double[fold_size][dimension];
			double current_mse = 0;
			for (int k = 0; k < 10; k++){
				double[][] test_fold_with_labels = folds.get(k);
				double[][] test_fold_data = BayesianLinearRegression.getDataFromDataWithLabels(test_fold_with_labels);
				double[] test_labels = BayesianLinearRegression.getLabelsFromDataWithLabels(test_fold_with_labels);
				
				double[][] train_fold_with_labels = BayesianLinearRegression.generateTrainingFolds(folds, k);
				double[][] train_fold_data = BayesianLinearRegression.getDataFromDataWithLabels(train_fold_with_labels);
				double[] train_labels = BayesianLinearRegression.getLabelsFromDataWithLabels(train_fold_with_labels);
				double[] current_w = BayesianLinearRegression.calculateW(lambda, train_fold_data, train_labels);
				current_mse += BayesianLinearRegression.mse(test_fold_data, current_w, test_labels);
			}
			double avg_mse = current_mse/(double)10;
			lambdas_mse[(int)lambda] = avg_mse;
		}

		//Arrays.sort(lambdas_mse);
	    double minValue = lambdas_mse[0];
	    double chosenLambda = 0;
	    for (int i = 1; i < lambdas_mse.length; i++) {
	        //System.out.println(i + " " + lambdas_mse[i]);
	    		if (lambdas_mse[i] < minValue) {
	            minValue = lambdas_mse[i];
	            chosenLambda = (double)i;
	        }
	    }
//	    System.out.println("~~~~~~~~~~~~~~~~~");
//	    System.out.println(chosenLambda);
		return chosenLambda;
	}
	
	public static HashMap<Integer, double[][]> splitFolds(double[][] data_with_labels) {
		int data_size = data_with_labels.length;
		int fold_size = data_size/10;
		List<double[]> list_data = Arrays.asList(data_with_labels); 
		HashMap<Integer, double[][]> folds = new HashMap<Integer, double[][]>();
		int index = 0;
		for(int i = 0; i < data_size; i = i + fold_size){
			int j = i + fold_size;
			if(j >= data_size){
				j = data_size;
			}
			
			List<double[]> temp_fold = list_data.subList(i, j);
			double[][] current_fold = temp_fold.toArray(new double[][]{});
			folds.put(index, current_fold);
			index++;
		}
		return folds;
	}
	
	public static double[][] generateTrainingFolds(HashMap<Integer, double[][]> folds, int k) {
		int dimension = folds.get(0)[0].length;
		List<double[]> train_data = new ArrayList<double[]>();
		for (int i = 0; i < 10; i++) {
			if (i != k) {
				//train_data.add(folds.get(i));
				train_data.addAll(Arrays.asList(folds.get(i)));
			}
		}
		double[][] train_folds = train_data.toArray(new double[][]{});
//		double[][] train_folds = new double[train_data.size()][dimension];
//		for (int j = 0; j < train_data.size(); j++) {
//			train_folds[j] = train_data.get(j);
//		}
		return train_folds;
	}
	
	public static double mse (double[][] phi, double[] w, double[] t) {
		double result = 0;
		double n = phi.length;
		for (int i = 0; i < n; i++) {
			double predict = 0;
			for (int j = 0; j < phi[i].length; j++) {
				predict += phi[i][j] * w[j];
			}
			result += Math.pow((predict - t[i]), 2);
		}
		return (result / n);
	}
	
	public static Boolean determineSingular (double lambda, double[][] phi) {
		Matrix phiMatrix = new Matrix(phi);
		int dimension = phiMatrix.transpose().getArray().length;
		Matrix identity = Matrix.identity(dimension, dimension);
		if ((identity.times(lambda).plus(phiMatrix.transpose().times(phiMatrix))).lu().isNonsingular()){
			return false;
		} else {
			return true;
		}
	}
	
	public static double[] calculateW (double lambda, double[][] phi, double[] t){
		Matrix phiMatrix = new Matrix(phi);
		int dimension = phiMatrix.transpose().getArray().length;
		Matrix identity = Matrix.identity(dimension, dimension);
		Matrix tMatrix = new Matrix(t, 1).transpose();
		Matrix w = (identity.times(lambda).plus(phiMatrix.transpose().times(phiMatrix))).inverse().times(phiMatrix.transpose()).times(tMatrix);
		return w.getColumnPackedCopy();
	}
	
	public static void printMatrix (Matrix m) {
		double[][] b = m.getArray();
		for(int i = 0; i < b.length; i++) {
		    for(int j = 0; j < b[i].length; j++) {        
		        System.out.print( " " + b[i][j] );
		    }
		    System.out.println("");
		}
	}
	
	public static void task3 () throws IOException {
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
	
	public static void main(String[] args) throws Exception{
		// TODO Auto-generated method stub
//		double[][] array = {{-1.,1.,0},{-4.,3.,0.},{1.,0.,2.}, {3, 4, 5}}; 
//		Matrix a = new Matrix(array); 
//		double[][] b = a.inverse().getArray();
//		String dataPath = "data/"; // data path
//		String trainingData = dataPath + "train-100-10.csv";
//		String trainingLabels = dataPath + "trainR-100-10.csv";
//		double[][] phi = BayesianLinearRegression.readData(trainingData);
//		double[] t = BayesianLinearRegression.readLabels(trainingLabels);
//		//Collections.shuffle(Arrays.asList(phi));
//		double[][] data_with_labels = BayesianLinearRegression.combineDataWithLabels(phi, t);
//		BayesianLinearRegression.implementTenFoldCross(data_with_labels);
////		
//		for (int i = 0; i < phi.length; i++) {
//			double[] data_label = new double [5];
//			for (int j = 0; j < phi[i].length; j++) {
//				data_label[j] = phi[i][j];
//			}
//			data_label[4] = t[i];
//			data_with_labels[i] = data_label;
//		}
		
//		double[] t = BayesianLinearRegression.readLabels(trainingLabels);
//		double[] w = BayesianLinearRegression.calculateW(0, phi, t);
//		double mse = BayesianLinearRegression.mse(phi, w, t);
		
		//BayesianLinearRegression.task1();
		
		// now loop through the rows of valsTransposed to print
//		for(int i = 0; i < w.length; i++) {       
//			System.out.println( " " + w[i]);
//		}
//		printMatrix(new Matrix(data_with_labels));
//		System.out.println(mse);
		//Task1.task1();
		
		//Task2.task2();
		BayesianLinearRegression.task3();
	}

}

