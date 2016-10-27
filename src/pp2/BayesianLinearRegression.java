package pp2;

import java.util.ArrayList;
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
	
	public static double[] calculateW (double lambda, double[][] phi, double[] t){
		int dimension = phi.length;
		Matrix identity = Matrix.identity(dimension, dimension);
		Matrix phiMatrix = new Matrix(phi);
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
	
	public static void main(String[] args) throws Exception{
		// TODO Auto-generated method stub
		double[][] array = {{-1.,1.,0},{-4.,3.,0.},{1.,0.,2.}, {3, 4, 5}}; 
		Matrix a = new Matrix(array); 
		double[][] b = a.inverse().getArray();
		String dataPath = "data/"; // data path
		String trainingData = dataPath + "haha.csv";
		String trainingLabels = dataPath + "hahaR.csv";
		double[][] phi = BayesianLinearRegression.readData(trainingData);
		double[] t = BayesianLinearRegression.readLabels(trainingLabels);
		double[] w = BayesianLinearRegression.calculateW(0, phi, t);
		double mse = BayesianLinearRegression.mse(phi, w, t);
		
		
		// now loop through the rows of valsTransposed to print
		for(int i = 0; i < w.length; i++) {       
			System.out.println( " " + w[i]);
		}
		//printMatrix(new Matrix(phi));
		System.out.println(mse);
	}

}

