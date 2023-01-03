import numpy as np
from sim.circuits import mux_1
from sim.PTM import bin_array, get_func_mat, B_mat
from sim.SEC import Ks_to_Mf, opt_K_max

def ptm_to_verilog(ptm, module_name, do_tb=True):
    n2, k2 = ptm.shape
    n = np.log2(n2).astype(np.int32)
    k = np.log2(k2).astype(np.int32)
    assert n <= 20 #Prevent large file outputs
    Bk = B_mat(k)
    Amat = ptm @ Bk
    fn = module_name + ".sv"
    with open(fn, 'w') as outfile:
        header_str = """module {}(
    input [{}:0] x,
    output logic [{}:0] z
);\n""".format(module_name, n-1, k-1)
        outfile.write(header_str)
        outfile.write("""always_comb \n\t case(x)\n""")
        for row in range(n2):
            row_b = bin_array(row, n)
            row_s = "\t\t{}'b{}: z = {}'b{}; \n".format(
                n, ''.join([str(x) for x in 1*row_b]), k, ''.join([str(1*a) for a in Amat[row, :]])
            )
            outfile.write(row_s)
        outfile.write("""\t endcase \nend \n""")
        outfile.write("endmodule")

    if do_tb:
        ptm_to_verilog_tb(ptm, module_name)

def ptm_to_verilog_tb(ptm, module_name):
    n2, k2 = ptm.shape
    n = np.log2(n2).astype(np.int32)
    k = np.log2(k2).astype(np.int32)
    assert n <= 20 #Prevent large file outputs
    Bk = B_mat(k)
    Amat = ptm @ Bk
    fn = module_name + "_tb.sv"
    with open(fn, 'w') as outfile:
        header_str = """`timescale 1ns/100ps
module {}_tb;
    logic [{}:0] x;
    logic [{}:0] z, z_correct;
    {} {}_dut(x, z);
    task check();
        if(z_correct !== z) begin
            $display("TESTCASE FAILED: z: %b, z_correct: %b", z, z_correct);
            $finish;
        end
    endtask

    initial begin
\n""".format(module_name, n-1, k-1, module_name, module_name)
        outfile.write(header_str)
        for row in range(n2):
            row_b = bin_array(row, n)
            row_s = "\t\tx={}'b{}; z_correct = {}'b{}; #5; check(); \n".format(
                n, ''.join([str(x) for x in 1*row_b]), k, ''.join([str(1*a) for a in Amat[row, :]])
            )
            outfile.write(row_s)
        footer_str = """
        $display("PASSED");
        $finish;
    end
endmodule"""
        outfile.write(footer_str)

def FA(a, b, cin):
    sum = np.bitwise_xor(np.bitwise_xor(a, b), cin)
    cout = np.bitwise_or(np.bitwise_or(
        np.bitwise_and(a, b),
        np.bitwise_and(a, cin)),
        np.bitwise_and(b, cin)
    )
    return sum, cout

def test_ptm_to_verilog():
    mux_ptm = get_func_mat(mux_1, 3, 1)
    ptm_to_verilog(mux_ptm, "mux")

    FA_ptm = get_func_mat(FA, 3, 2)
    ptm_to_verilog(FA_ptm, "FA")

def test_ptm_to_tb():
    FA_ptm = get_func_mat(FA, 3, 2)
    ptm_to_verilog_tb(FA_ptm, "FA")

def test_larger_ptm_to_verilog():
    gb4_ptm = np.load("gb4_ptm.npy")
    ptm_to_verilog_tb(gb4_ptm, "gb4") #Already did this one

    A = gb4_ptm @ B_mat(4)
    Ks = []
    Ks_opt = []
    for i in range(4):
        K = A[:, i].reshape(2**4, 2**16).T
        K_opt = opt_K_max(K)
        Ks.append(K)
        Ks_opt.append(K_opt)
    gb4_ptm_opt = Ks_to_Mf(Ks_opt)
    ptm_to_verilog_tb(gb4_ptm_opt, "gb4_opt")