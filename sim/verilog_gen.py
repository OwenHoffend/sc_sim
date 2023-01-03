import numpy as np
from sim.PTM import bin_array, B_mat

def ptm_to_verilog(ptm, module_name):
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
        outfile.write("""always_comb begin \n\t case(x)\n""")
        for row in range(n2):
            row_b = bin_array(row, n)
            row_s = "\t\t{}'b{}: z = {}'b{}; \n".format(
                n, ''.join([str(x) for x in 1*row_b]), k, ''.join([str(1*a) for a in Amat[row, :]])
            )
            outfile.write(row_s)
        outfile.write("""\t endcase \nend \n""")
        outfile.write("endmodule")

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
            $display("TESTCASE FAILED: x: %b, z: %b, z_correct: %b", x, z, z_correct);
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

def espresso_out_to_verilog(ifn, module_name):
    ofn = module_name + ".sv"
    with open(ifn) as infile:
        with open(ofn, 'w') as outfile:
            n = int(infile.readline().split(' ')[1])
            k = int(infile.readline().split(' ')[1])
            _ = infile.readline() #don't need the third entry
            line = infile.readline()
            header_str = """module {}(
    input [{}:0] x,
    output logic [{}:0] z
);\n""".format(module_name, n-1, k-1)
            outfile.write(header_str)
            outfile.write("""always_comb begin \n\t z = 4'b0000;\n""")
            while not line.startswith('.'):
                instr, outstr = line.split(' ')
                instr = instr.replace('-', '?')
                outstr = outstr.replace('\n', '')
                row_s = "\tif(x==?{}'b{}) z |= {}'b{}; \n".format(n, instr, k, outstr)
                outfile.write(row_s)
                line = infile.readline()
            outfile.write("end \n".format(k, k*'0'))
            outfile.write("endmodule")