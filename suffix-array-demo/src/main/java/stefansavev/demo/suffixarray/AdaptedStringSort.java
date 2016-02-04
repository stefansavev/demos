package stefansavev.demo.suffixarray;

//based on the following code:
//http://www.cs.princeton.edu/~rs/strings/demo.c
//http://www.codeproject.com/Articles/146086/Fast-String-Sort-in-C-and-F
public class AdaptedStringSort {
    static int stringLength(int[] input, int s) {
        int len = (input.length - s);
        return Math.min(25, len); //putting a limit of 25, this means we can search for phrases upto length 25
    }

    static int charAt(int[] input, int offset, int pos) {
        return input[offset + pos];
    }

    static int charOrNull(int[] input, int s, int pos) {
        if (pos >= stringLength(input, s))
            return 0;
        return charAt(input, s, pos);
    }

    static int medianOf3(int[] input, int[] x, int a, int b, int c, int depth) {
        int va, vb, vc;
        if ((va = charOrNull(input, x[a], depth)) == (vb = charOrNull(input, x[b], depth)))
            return a;
        if ((vc = charOrNull(input, x[c], depth)) == va || vc == vb)
            return c;
        return va < vb ?
                (vb < vc ? b : (va < vc ? c : a))
                : (vb > vc ? b : (va < vc ? a : c));
    }

    //Pathological case is: strings with long common prefixes will
    //          cause long running times
    public static void insertionSort(int[] input, int[] x, int a, int n, final int depth) {
        int pi;
        int pj;
        for (pi = a + 1; --n > 0; pi++) {
            for (pj = pi; pj > a; pj--) {
                int s = x[pj - 1];
                int t = x[pj];
                int d = depth;

                int s_len = stringLength(input, s);
                int t_len = stringLength(input, t);
                while (d < s_len && d < t_len && charAt(input, s, d) == charAt(input, t, d)) {
                    d++;
                }
                ;
                if (d == s_len || (d < t_len && charAt(input, s, d) <= charAt(input, t, d)))
                    break;
                int pj1 = pj - 1;
                int tmp = x[pj];
                x[pj] = x[pj1];
                x[pj1] = tmp;
            }
        }

    }

    static void vecswap(int[] x, int a, int b, long n) {
        while (n-- > 0) {
            int t = x[a];
            x[a++] = x[b];
            x[b++] = t;
        }
    }

    //input: an array of words represented by integer ids
    //output suffix array represented by offset into the original array
    public static int[] sort(int[] input) {
        int[] copy = new int[input.length];
        for (int i = 0; i < input.length; i++) {
            if (input[i] < 1) {
                throw new IllegalStateException("Accepted indexes have to be >= 1");
            }
            copy[i] = i;
        }

        inPlaceSort(input, copy, 0, copy.length, 0);
        return copy;
    }


    static void inPlaceSort(int[] input, int[] x, int a, int n, int depth) {
        int partval;
        int d, r;
        int pa;
        int pb;
        int pc;
        int pd;
        int pl;
        int pm;
        int pn;
        int t;

        if (n < 10) {
            insertionSort(input, x, a, n, depth);
            return;
        }
        pl = a;
        pm = a + n / 2;
        pn = a + (n - 1);
        if (n > 30) {
            // On big arrays, pseudomedian of 9
            d = (n / 8);
            pl = medianOf3(input, x, pl, pl + d, pl + 2 * d, depth);
            pm = medianOf3(input, x, pm - d, pm, pm + d, depth);
            pn = medianOf3(input, x, pn - 2 * d, pn - d, pn, depth);
        }
        pm = medianOf3(input, x, pl, pm, pn, depth);

        {
            t = x[a];
            x[a] = x[pm];
            x[pm] = t;
        }

        pa = pb = a + 1;
        pc = pd = a + n - 1;

        partval = charOrNull(input, x[a], depth);
        int len = stringLength(input, x[a]);
        boolean empty = (len == depth);

        for (; ; ) {
            while (pb <= pc && (r = (empty ? (stringLength(input, x[pb]) - depth) : ((depth == stringLength(input, x[pb])) ? -1 : (charAt(input, x[pb], depth) - partval)))) <= 0) {
                if (r == 0) {
                    //swap(pa, pb);
                    {
                        t = x[pa];
                        x[pa] = x[pb];
                        x[pb] = t;
                    }
                    pa++;
                }
                pb++;
            }
            while (pb <= pc && (r = (empty ? (stringLength(input, x[pc]) - depth) : ((depth == stringLength(input, x[pc])) ? -1 : (charAt(input, x[pc], depth) - partval)))) >= 0) {
                if (r == 0) {   //swap(pc, pd);
                    {
                        t = x[pc];
                        x[pc] = x[pd];
                        x[pd] = t;
                    }
                    pd--;
                }
                pc--;
            }
            if (pb > pc) break;

            //swap(pb, pc);
            {
                t = x[pb];
                x[pb] = x[pc];
                x[pc] = t;
            }
            pb++;
            pc--;
        }

        pn = a + n;
        if (pa - a < pb - pa) {
            r = (pa - a);
        } else {
            r = (pb - pa);
        }

        //swapping pointers to strings
        vecswap(x, a, pb - r, r);
        if (pd - pc < pn - pd - 1) {
            r = pd - pc;
        } else {
            r = pn - pd - 1;
        }
        vecswap(x, pb, pn - r, r);
        r = pb - pa;
        if (pa - a + pn - pd - 1 > 1 && stringLength(input, x[a + r]) > depth) //by definition x[a + r] has at least one element
            inPlaceSort(input, x, a + r, pa - a + pn - pd - 1, depth + 1);
        if (r > 1)
            inPlaceSort(input, x, a, r, depth);
        if ((r = pd - pc) > 1)
            inPlaceSort(input, x, a + n - r, r, depth);
    }
}

