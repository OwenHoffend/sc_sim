from email import header
import numpy as np
from sim.PTM import bin_array

def ptm_to_verilog(ptm, module_name):
    nv2, nc2 = ptm.shape
    n = np.log2(nv2).astype(np.int)
    k = np.log2(nc2).astype(np.int)
    fn = module_name + ".v"
    with open(fn, 'w') as outfile:
        header_str = """module {}(
            input [{}-1:0] x,
            output [{}-1:0] z
        );
        """.format(module_name, n, k)
        outfile.write(header_str)
        outfile.write("""always @* begin case(x)""")
        for row in range(nv2):
            row_b = bin_array(row, n)
            row_s = "{}'b{}: z = {};".format(
                n, row_b, ''.join([str(a) for a in ptm[row, :]])
            )
            outfile.write(row_s)
        outfile.write("""endcase end""")
        outfile.write("endmodule")