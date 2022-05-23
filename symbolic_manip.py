import re
import functools
import copy
import numpy as np

import sim.corr_preservation as cp
import sim.PTM as pm

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

    def _dup_term_reduce(self, terms, coeffs):
        new = {}
        skips = []
        for i in range(len(terms)):
            if i in skips:
                continue
            term1 = terms[i]
            c1 = coeffs[i]
            var_set1 = set(re.findall(r"[^\*\s]+(?=[\*])*", term1))
            for j in range(i+1, len(terms)):
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
        self._dup_term_reduce(terms, coeffs)

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
        for term, coeff in self.poly.items():
            if not first and float(coeff) < 0.0:
                strrep = strrep[:-1] #Trim the '+' symbol
            strrep += "{}({})+".format(coeff, term)
            first = False
        strrep = strrep[:-1]
        return strrep

    #Substitute all instances of a variable with a scalar, then reduce
    def sub_scalar(self, var_str, scalar):
        terms = []
        coeffs = []
        for term, coeff in self.poly.items():
            if var_str in term:
                final_term = re.sub(r"{}\^[\d\.]+[\*]{}".format(var_str, "{0,1}"), '', term)
                #.format() eats '{' '}' characters, so the second sub is a hack to get those into the regex again
                terms.append(re.sub(r"\*$", '', final_term))
                _exp = re.findall(r"(?<={}[\^])[\d\.]+".format(var_str), term)
                assert len(_exp) == 1
                coeffs.append(str(float(coeff) * (scalar ** int(_exp[0]))))
            else:
                terms.append(term)
                coeffs.append(coeff)
        if terms == ['']:
            terms = ['@^1']
        self._dup_term_reduce(terms, coeffs)

    def get_latex(self):
        latex_str = ""
        for term, coeff in self.poly.items():
            _term = re.sub(r"\*|\(|\)|\@|\^1|\.0", '', term) #Remove extra syntax
            _term = re.sub(r"(?<=[a-zA-Z])[\d]+", r"_{\g<0>}", _term) #Add subscripting to variables
            _coeff = re.sub(r"\.0", '', coeff)
            if float(coeff) < 0:
                latex_str = latex_str[:-1]
            if abs(float(coeff)) == 1 and _term != '':
                _coeff = _coeff[:-1]
            latex_str += _coeff + _term + "+"
        latex_str = latex_str[:-1]
        return latex_str

def scalar_mat_poly(mat):
    m, n = mat.shape
    out = np.zeros((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            out[i, j] = Polynomial(poly_string="{}(@^1)".format(mat[i,j]))
    return out

def vin_poly(nvars, vname='v'):
    dim = 2 ** nvars
    vec = np.array([Polynomial(poly_string="1.0({}{}^1)".format(vname, i)) for i in range(dim)], dtype=object)
    return np.expand_dims(vec, axis=1)

def vin_poly_bernoulli_mc0(nvars):
    dim = 2 ** nvars
    Bn = cp.B_mat(nvars)
    vin = np.zeros((dim,), dtype=object)
    pos_polys = [Polynomial(poly_string="1.0(p{}^1)".format(i)) for i in range(nvars)]
    neg_polys = [Polynomial(poly_string="1.0(@^1)-1.0(p{}^1)".format(i)) for i in range(nvars)]
    for i in range(dim):
        row_poly = [pos_polys[j] if Bn[i][j] else neg_polys[j] for j in range(nvars)]
        vin[i] = functools.reduce(lambda a, b: a*b, row_poly)
    return vin

def vin_poly_bernoulli_mc1(nvars, ordering=None, names=None):
    dim = 2 ** nvars
    vin = np.array([Polynomial(poly_string="0.0(@^1)") for _ in range(dim)])
    if names == None:
        name_arr = ["p{}".format(i) for i in range(nvars)]
    else:
        name_arr = names
    if ordering is None:
        ordering = list(range(nvars))
    polys = [Polynomial(poly_string="1.0({}^1)".format(name_arr[i])) for i in range(nvars)]
    vin[0] = Polynomial(poly_string="1.0(@^1)") - polys[ordering[0]]
    vin[dim-1] = polys[ordering[nvars-1]]
    i = 0
    for k in range(1, nvars):
        i += 2 ** ordering[k-1]
        vin[i] = polys[ordering[k-1]] - polys[ordering[k]]
    return vin

def vin_covmat_bernoulli(nvars, corr=0, custom=None):
    dim = 2 ** nvars
    mat = np.zeros((dim, dim), dtype=object)
    if custom is not None:
        vin = custom
    else:
        if corr:
            vin = vin_poly_bernoulli_mc1(nvars)
        else:
            vin = vin_poly_bernoulli_mc0(nvars)
    _one = Polynomial(poly_string="1.0(@^1)")
    _zero = Polynomial(poly_string="0.0(@^1)")
    for i in range(dim):
        for j in range(dim):
            if i == j:
                mat[i, i] = vin[i] * (_one - vin[i])
            else:
                mat[i, j] = vin[i] * (_zero - vin[j])
    return mat

def mat_sub_scalar(np_mat, var_str, scalar):
    m, n = np_mat.shape
    for i in range(m):
        for j in range(n):
            np_mat[i,j].sub_scalar(var_str, scalar)
    return np_mat

def mat_to_latex(np_mat):
    m, n = np_mat.shape
    latex_str = "\\begin{bmatrix}"
    for i in range(m):
        for j in range(n):
            latex_str += np_mat[i,j].get_latex()
            if j < n - 1:
                latex_str += ' & '
        if i < m-1:
            latex_str += "\\\\"
    latex_str += "\\end{bmatrix}"
    return latex_str

def symbolic_cov_mat_bernoulli(Mf, num_inputs, num_ouputs, corr=0, custom=None):
    Bk = pm.B_mat(num_ouputs)
    vin_mat = vin_covmat_bernoulli(num_inputs, corr=corr, custom=custom)
    A_mat = scalar_mat_poly((Mf @ Bk) * 1)
    return A_mat.T @ vin_mat @ A_mat