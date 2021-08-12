import re
import functools
import copy
import numpy as np

import sim.corr_preservation as cp

class Polynomial:
    def _flstr_add(self, a, b):
        return str(float(a) + float(b))

    def _istr_add(self, a, b):
        return str(int(a) + int(b))

    def _flstr_mul(self, a, b):
        return str(float(a) * float(b))

    def _merge_kv(self, keys, vals):
        assert len(keys) == len(vals)
        return {k : v for k, v in zip(keys, vals)}

    def __init__(self, **kwargs):
        if "poly_string" in kwargs:
            poly_string = kwargs["poly_string"]
            terms = re.findall(r"(?<=[(])[^)]*(?=[)])", poly_string)
            coeffs = re.findall(r"[-\d\.]+(?=[(])", poly_string)

            assert len(terms) == len(coeffs)
            self.poly = self._merge_kv(terms, coeffs)
        if "poly" in kwargs:
            self.poly = kwargs["poly"]
            terms = list(self.poly.keys())
            coeffs = list(self.poly.values())

        #Duplicate term reduction
        new = {}
        skips = []
        for i in range(len(self.poly)):
            if i in skips:
                continue
            term1 = terms[i]
            c1 = coeffs[i]
            var_set1 = set(re.findall(r"[^\*\s]+(?=[\*])*", term1))
            for j in range(i+1, len(self.poly)):
                term2 = terms[j]
                c2 = coeffs[j]
                var_set2 = set(re.findall(r"[^\*\s]+(?=[\*])*", term2))
                if var_set1 == var_set2:
                    skips.append(j)
                    c1 = self._flstr_add(c1, c2)
            if float(c1) != 0:
                new[term1] = c1
            if new == {}: #Gaurd against empty polynomials
                new["@^1"] = "0.0"
        self.poly = new

    def __add__(self, other):
        new = copy.copy(self.poly)
        for k, v in other.poly.items():
            if k in new:
                new[k] = self._flstr_add(v, new[k])
            else:
                new[k] = v
        return Polynomial(poly=new)

    def __sub__(self, other):
        other_neg = copy.copy(other) * Polynomial(poly_string="-1.0(@^1)")
        return self.__add__(other_neg)

    def __mul__(self, other):
        new = {}
        for term1, c1 in self.poly.items():
            vars1 = re.findall(r"[^*\W]+(?=[\^][\d]+)|@", term1)
            exps1 = re.findall(r"(?<=[\^])[\d\.]*", term1)
            vdict1 = self._merge_kv(vars1, exps1)
            for term2, c2 in other.poly.items():
                coeff_mul = float(c1) * float(c2)
                if coeff_mul == 0:
                    continue
                coeff_mul = str(coeff_mul)
                vars2 = re.findall(r"[^*\W]+(?=[\^][\d]+)|@", term2)
                exps2 = re.findall(r"(?<=[\^])[\d\.]*", term2)

                merge_dict = copy.copy(vdict1)
                for var, exp in zip(vars2, exps2):
                    if var in merge_dict:
                        merge_dict[var] = self._istr_add(merge_dict[var], exp)
                    else:
                        merge_dict[var] = exp

                merged_var = ""
                for var, exp in merge_dict.items():
                    if var == "@":
                        continue
                    else:
                        merged_var += var + "^" + str(exp) + "*"
                if merged_var == "":
                    merged_var = "@^1"
                else:
                    merged_var = merged_var[:-1]
                if merged_var in new:
                    new[merged_var] = self._flstr_add(new[merged_var], coeff_mul)
                else:
                    new[merged_var] = coeff_mul
        return Polynomial(poly=new)

    def __pow__(self, other):
        if other > 1:
            return functools.reduce(lambda a, b: a * b, [self for _ in range(other)])
        return copy.copy(self)

    def __repr__(self):
        strrep = ""
        first = True
        for var, coeff in self.poly.items():
            if not first and float(coeff) < 0.0:
                strrep = strrep[:-1] #Trim the '+' symbol
            strrep += "{}({})+".format(coeff, var)
            first = False
        strrep = strrep[:-1]
        return strrep

def scalar_mat_poly(mat):
    m, n = mat.shape
    out = np.zeros((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            out[i, j] = Polynomial(poly_string="{}(@^1)".format(mat[i,j]))
    return out

def vin_poly_bernoulli(nvars):
    dim = 2 ** nvars
    Bn = cp.B_mat(nvars)
    vin = np.zeros((dim,), dtype=object)
    pos_polys = [Polynomial(poly_string="1.0(p{}^1)".format(i)) for i in range(nvars)]
    neg_polys = [Polynomial(poly_string="1.0(@^1)-1.0(p{}^1)".format(i)) for i in range(nvars)]
    for i in range(dim):
        row_poly = [pos_polys[j] if Bn[i][j] else neg_polys[j] for j in range(nvars)]
        vin[i] = functools.reduce(lambda a, b: a*b, row_poly)
    return vin

def vin_covmat_bernoulli(nvars):
    dim = 2 ** nvars
    mat = np.zeros((dim, dim), dtype=object)
    vin = vin_poly_bernoulli(nvars)
    _one = Polynomial(poly_string="1.0(@^1)")
    _zero = Polynomial(poly_string="0.0(@^1)")
    for i in range(dim):
        for j in range(dim):
            if i == j:
                mat[i, i] = vin[i] * (_one - vin[i])
            else:
                mat[i, j] = vin[i] * (_zero - vin[j])
    return mat

if __name__ == "__main__":
    pass
    #Polynomial creation test:
    #test_poly = Polynomial(poly_string="0.5(x1^1*x2^2)-0.5(x2^2*x1^1)+1.0(x2^2)")
    #print(test_poly.poly)

    #Sum test:
    #test_poly = Polynomial(poly_string="0.5(x1^2*x2^2)-1.0(x1^1*x2^1)+1(x1^2*x2^2)+1(@)")
    #test_poly2 = Polynomial(poly_string="0.3(x1^1*x2^1)-5(@)")
    #res = test_poly + test_poly2

    #Sub test:
    #test_poly = Polynomial(poly_string="0.5(x1^1)+0.3(x1^2*x2^1)")
    #test_poly2 = Polynomial(poly_string="0.1(x1^1)-5.0(@^1)")
    #print(test_poly - test_poly2)

    #Mul test:
    #test_poly = Polynomial(poly_string="0.5(a^1)+2(@^1)")
    #test_poly2 = Polynomial(poly_string="0.25(a^2*b^1)+0.5(a^1)+6(@^1)")
    #res = test_poly2 ** 2
    #print(test_poly.poly)
    #print(test_poly2.poly)
    #print(res.poly)

    #Second mul test:
    #test_poly = Polynomial(poly_string="1.0(a^1)+1.0(b^1)")
    #test_poly2 = Polynomial(poly_string="1.0(b^1)+1.0(a^1)")
    #res = test_poly * test_poly2
    #print(res.poly)

    #Numpy multiplication test
    #test_vec = np.array([[test_poly, test_poly2]], dtype=object)
    #result = test_vec @ test_vec.T
    #print(result[0,0].poly)

    #vin_poly_bernoulli test
    #mat = vin_poly_bernoulli(4)
    #print(mat)

    #vin_covmat_bernoulli test
    #mat = vin_covmat_bernoulli(2)
    #print(mat)

    #Test matrix product
    #mat = np.array([
    #    [1.0, 0],
    #    [0.5, -0.3]
    #])
    #test = scalar_mat_poly(mat) @ vin_poly_bernoulli(1)
    #print(test)