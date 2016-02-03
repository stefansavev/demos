package sais;

public class SuffixArraySort {

    public static int[] sort(int[] T, int k){
        int n = T.length;
        int[] SA = new int[n];

        new Sais().suffixsort(T, SA, n, k);
        return SA;
    }
}
