package pp2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.io.*;
import java.util.List;
import Jama.Matrix;

/**
 * The main class for implementing project 2
 * @author zhaokunxue
 *
 */
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
	
	/**
	 * The function to read label file
	 * @param filename label file name
	 * @return an array contains all labels
	 * @throws IOException
	 */
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
                t[i] = labels.get(i);
             }
            br.close();
            
        } catch (FileNotFoundException fnfe) {
            System.err.println("file not found");
        }
		return t;
	}
	
	/**
	 * get the first n labels from the labels array we get when we read the labels file
	 * @param labels an array of labels
	 * @param n the first n labels we need 
	 * @return the first n labels from the labels array
	 */
	public static double[] getFirstNLabels(double[] labels, int n) {
		double[] result = new double[n];
		for (int i = 0; i < n; i++) {
			result[i] = labels[i];
		}
		return result;
	}
	
	/**
	 * get the first n data from the data matrix we have
	 * @param data the data matrix we have
	 * @param n the first data we want
	 * @return the first n data from the given data matrix
	 */
	public static double[][] getFirstNData(double[][] data, int n) {
		int dimention = data[0].length;
		double[][] results = new double[n][dimention];
		for (int i = 0; i < n; i++){
			results[i] = data[i];
		}
		return results;
	}

	/**
	 * Combine data with labels
	 * @param data the data matrix
	 * @param labels the label vector
	 * @return a matrix by adding the label vector as the last column to the data matrix
	 */
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
	
	/**
	 * Extract data from data with labels matrix
	 * @param data_with_labels a matrix combined by data and labels
	 * @return the data matrix from the data_with_labels array
	 */
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
	
	/**
	 * Extract labels from data with labels matrix
	 * @param data_with_labels a matrix combined by data and labels
	 * @return the labeles vector from the data_with_labels matrix
	 */
	public static double[] getLabelsFromDataWithLabels(double[][] data_with_labels) {
		double[] labels = new double[data_with_labels.length];
		int dimension = data_with_labels[0].length;
		for (int i = 0; i < data_with_labels.length; i++) {
			labels[i] = data_with_labels[i][dimension-1];
		}
		return labels;
	}
	
	/**
	 * The function implements 10fold cross validation
	 * @param data_with_labels a matrix combined by data and labels
	 * @return the minimum mse calculated from 10fole cross validation
	 */
	public static double implementTenFoldCross(double[][] data_with_labels) {
		double[] lambdas_mse = new double[151];
		HashMap<Integer, double[][]> folds = BayesianLinearRegression.splitFolds(data_with_labels);
		// iterate for lambda
		for (double lambda = 0; lambda < 151; lambda++) {
			double current_mse = 0;
			// iterate for each fold
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
	    double[] minMse = findMinMse(lambdas_mse);
		return minMse[0];
	}
	
	/**
	 * The function splits the given data with label matrix
	 * @param data_with_labels a matrix combined with data and labels
	 * @return a hashmap corresponding to each fold
	 */
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
	
	/**
	 * Generate the training folds for cross validation
	 * @param folds the folds we have for cross validation
	 * @param k current test fold
	 * @return a matrix using for training folds
	 */
	public static double[][] generateTrainingFolds(HashMap<Integer, double[][]> folds, int k) {
		List<double[]> train_data = new ArrayList<double[]>();
		for (int i = 0; i < 10; i++) {
			if (i != k) {
				train_data.addAll(Arrays.asList(folds.get(i)));
			}
		}
		double[][] train_folds = train_data.toArray(new double[][]{});
		return train_folds;
	}
	
	/**
	 * The function calculates the mse
	 * @param phi the data matrix
	 * @param w the weighted parameter
	 * @param t the labels
	 * @return the mse
	 */
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
	
	/**
	 * Calculate M_n for Bayesian Model Selection
	 * @param phi the data matrix
	 * @param t the label vector
	 * @return MAP M_n for Bayesian Model Selection
	 */
	public static double[] calculateMn(double[][] phi, double[] t) {
		int n = phi.length;
		int dimension = phi[0].length;
		double alpha = 1;
		double beta = 1;
		
		Matrix phiMatrix = new Matrix(phi);
		Matrix tMatrix = new Matrix(t, 1).transpose();
		Matrix identity = Matrix.identity(dimension, dimension);
		double alpha_changes = 100;
		double beta_changes = 100;
		double delta = Math.pow(10, -4);

		Matrix sn = (identity.times(alpha).plus(phiMatrix.transpose().times(phiMatrix).times(beta))).inverse();
		Matrix mn = sn.times(phiMatrix.transpose()).times(tMatrix).times(beta);

		while (Math.abs(alpha_changes) > delta || Math.abs(beta_changes) > delta) {
			double[] eigenvalues = phiMatrix.transpose().times(phiMatrix).times(beta).eig().getRealEigenvalues();
			
			double gamma = 0;
			for (double eigenvalue : eigenvalues) {
				gamma += eigenvalue /(alpha + eigenvalue);
			}	
			
			double update_alpha = gamma / (mn.transpose().times(mn)).get(0,0);
			Matrix temp = phiMatrix.times(mn).minus(tMatrix);
			double[] temp_array = temp.getColumnPackedCopy();
			double norm = 0;
			for (double ta : temp_array) {
				norm += Math.pow(ta, 2);
			}

			double update_beta = (n - gamma)/norm;
			
			sn = (identity.times(update_alpha).plus(phiMatrix.transpose().times(phiMatrix).times(update_beta))).inverse();
			mn = sn.times(phiMatrix.transpose()).times(tMatrix).times(update_beta);
			alpha_changes = (update_alpha - alpha)/alpha;
			beta_changes = (update_beta - beta)/beta;
			alpha = update_alpha;
			beta = update_beta;
		}
		
		return mn.getColumnPackedCopy();
	}
	
	/**
	 * Calculate the weighted parameters vector w
	 * @param lambda the lambda value
	 * @param phi the data matrix
	 * @param t the label vector
	 * @return the weighte parameters w
	 */
	public static double[] calculateW (double lambda, double[][] phi, double[] t){
		Matrix phiMatrix = new Matrix(phi);
		int dimension = phiMatrix.transpose().getArray().length;
		Matrix identity = Matrix.identity(dimension, dimension);
		Matrix tMatrix = new Matrix(t, 1).transpose();
		Matrix w = (identity.times(lambda).plus(phiMatrix.transpose().times(phiMatrix))).inverse().times(phiMatrix.transpose()).times(tMatrix);
		return w.getColumnPackedCopy();
	}

	/**
	 * Find the minimum mse in an array of mses
	 * @param lambdas_mse the array of mses
	 * @return the minimum mse in the given array
	 */
	public static double[] findMinMse (double[] lambdas_mse) {
		double results[] = new double[2];
	    double minValue = lambdas_mse[0];
	    double chosenLambda = 0;
	    for (int i = 1; i < lambdas_mse.length; i++) {
	    		if (lambdas_mse[i] < minValue) {
	            minValue = lambdas_mse[i];
	            chosenLambda = (double)i;
	        }
	    }
	    results[0] = chosenLambda;
	    results[1] = minValue;
	    return results;
	}
	
	/**
	 * Determine whether the matrix a singular
	 * @param lambda lambda value
	 * @param phi the data matrix
	 * @return true if the matrix is singular, else false
	 */
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
	
	
	/**
	 * The main function for implementing task1 to task 4
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception{
		Task1.task1();
		Task2.task2();
		Task3.task3();
		Task4.task4();
	}

}

