import numpy as np
from sim.circuits import mux_1
from sim.PTM import bin_array, get_func_mat, B_mat

def ptm_to_verilog(ptm, module_name):
    nv2, nc2 = ptm.shape
    n = np.log2(nv2).astype(np.int32)
    k = np.log2(nc2).astype(np.int32)
    Bk = B_mat(k)
    Amat = ptm @ Bk
    fn = module_name + ".v"
    with open(fn, 'w') as outfile:
        header_str = """module {}(
    input [{}:0] x,
    output reg [{}:0] z
);\n""".format(module_name, n-1, k-1)
        outfile.write(header_str)
        outfile.write("""always @* begin \n\t case(x)\n""")
        for row in range(nv2):
            row_b = bin_array(row, n)
            row_s = "\t\t{}'b{}: z = {}'b{}; \n".format(
                n, ''.join([str(x) for x in 1*row_b]), k, ''.join([str(1*a) for a in Amat[row, :]])
            )
            outfile.write(row_s)
        outfile.write("""\t endcase \nend \n""")
        outfile.write("endmodule")

def test_ptm_to_verilog():
    mux_ptm = get_func_mat(mux_1, 3, 1)
    ptm_to_verilog(mux_ptm, "mux")

    def FA(a, b, cin):
        sum = np.bitwise_xor(np.bitwise_xor(a, b), cin)
        cout = np.bitwise_or(np.bitwise_or(
            np.bitwise_and(a, b),
            np.bitwise_and(a, cin)),
            np.bitwise_and(b, cin)
        )
        return sum, cout
    FA_ptm = get_func_mat(FA, 3, 2)
    ptm_to_verilog(FA_ptm, "FA")