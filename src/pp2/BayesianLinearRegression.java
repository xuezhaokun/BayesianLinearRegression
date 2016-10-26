package pp2;

import java.util.ArrayList;
import java.io.*;
import java.util.List;
import Jama.Matrix;

public class BayesianLinearRegression {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		double[][] array = {{-1.,1.,0},{-4.,3.,0.},{1.,0.,2.}}; 
		Matrix a = new Matrix(array); 
		double[][] b = a.inverse().getArray();

		// now loop through the rows of valsTransposed to print
		for(int i = 0; i < b.length; i++) {
		    for(int j = 0; j < b[i].length; j++) {        
		        System.out.print( " " + b[i][j] );
		    }
		}
		//System.out.println(a.toString());
		//System.out.println(a.inverse().getArray());
	}

}

