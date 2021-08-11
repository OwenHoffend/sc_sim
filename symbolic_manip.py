import re
import copy

class Polynomial:
    def _flstr_add(self, a, b):
        return str(float(a) + float(b))

    def _flstr_mul(self, a, b):
        return str(float(a) * float(b))

    def _merge_kv(self, keys, vals, duplicate_func=None):
        _dict = {}
        assert len(keys) == len(vals)
        for k, v in zip(keys, vals):
            if duplicate_func is not None and v in _dict:
                _dict[k] = duplicate_func(_dict[k], v)
            else:
                _dict[k] = v
        return _dict

    def __init__(self, **kwargs):
        """Initialize the polynomial based on the poly_string of form: coeff(name)^power"""
        if "poly_string" in kwargs:
            poly_string = kwargs["poly_string"]
            vars = re.findall(r"(?<=[(])[^)]*(?=[)])", poly_string)
            coeffs = re.findall(r"[-\d\.]+(?=[(])", poly_string)

            assert len(vars) == len(coeffs)
            self.poly = self._merge_kv(vars, coeffs, duplicate_func=self._flstr_add)
        if "poly" in kwargs:
            self.poly = kwargs["poly"]

    def __add__(self, other):
        new = copy.copy(self.poly)
        for k, v in other.poly.items():
            if k in new:
                new[k] = self._flstr_add(v, new[k])
            else:
                new[k] = v
        return Polynomial(poly=new)

    def __mul__(self, other):
        new = {}
        for k1, c1 in self.poly.items():
            vars1 = re.findall(r"[^*\W]+(?=[\^][\d]+)|@", k1)
            exps1 = re.findall(r"(?<=[\^])[\d\.]*", k1)
            vdict1 = self._merge_kv(vars1, exps1)
            for k2, c2 in other.poly.items():
                vars2 = re.findall(r"[^*\W]+(?=[\^][\d]+)|@", k2)
                exps2 = re.findall(r"(?<=[\^])[\d\.]*", k2)

                merge_dict = copy.copy(vdict1)
                for var, exp in zip(vars2, exps2):
                    if var in merge_dict:
                        merge_dict[var] = self._flstr_add(merge_dict[var], exp)
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
                    new[merged_var] = self._flstr_add(new[merged_var], self._flstr_mul(c1, c2))
                else:
                    new[merged_var] = self._flstr_mul(c1, c2)
        return Polynomial(poly=new)

if __name__ == "__main__":

    #Sum test:
    #test_poly = Polynomial(poly_string="0.5(x1^2*x2^2)-1.0(x1^1*x2^1)+1(x1^2*x2^2)+1(@)")
    #test_poly2 = Polynomial(poly_string="0.3(x1^1*x2^1)-5(@)")
    #res = test_poly + test_poly2

    #Mul test:
    test_poly = Polynomial(poly_string="0.5(x1^1)+2(@^1)")
    test_poly2 = Polynomial(poly_string="0.25(x1^2*x2^1)+0.5(x1^1)+6(@^1)")
    res = test_poly * test_poly2
    print(test_poly.poly)
    print(test_poly2.poly)
    print(res.poly)
